# Based on https://github.com/youtube/api-samples/blob/master/python/upload_video.py

import http.client as httplib
import httplib2
import random
import time
import os
import logging

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow


# Explicitly tell the underlying HTTP transport library not to retry, since
# we are handling retry logic ourselves.
httplib2.RETRIES = 1

# Maximum number of times to retry before giving up.
MAX_RETRIES = 10

# Always retry when these exceptions are raised.
RETRIABLE_EXCEPTIONS = (httplib2.HttpLib2Error, IOError, httplib.NotConnected,
  httplib.IncompleteRead, httplib.ImproperConnectionState,
  httplib.CannotSendRequest, httplib.CannotSendHeader,
  httplib.ResponseNotReady, httplib.BadStatusLine)

# Always retry when an apiclient.errors.HttpError with one of these status
# codes is raised.
RETRIABLE_STATUS_CODES = [500, 502, 503, 504]

# The CLIENT_SECRETS_FILE variable specifies the name of a file that contains
# the OAuth 2.0 information for this application, including its client_id and
# client_secret. You can acquire an OAuth 2.0 client ID and client secret from
# the {{ Google Cloud Console }} at
# {{ https://cloud.google.com/console }}.
# Please ensure that you have enabled the YouTube Data API for your project.
# For more information about using OAuth2 to access the YouTube Data API, see:
#   https://developers.google.com/youtube/v3/guides/authentication
# For more information about the client_secrets.json file format, see:
#   https://developers.google.com/api-client-library/python/guide/aaa_client_secrets
CLIENT_SECRETS_FILE = 'client_secret.json'

# This OAuth 2.0 access scope allows an application to upload files to the
# authenticated user's YouTube channel, but doesn't allow other types of access.
SCOPES = ['https://www.googleapis.com/auth/youtube.upload']
API_SERVICE_NAME = 'youtube'
API_VERSION = 'v3'

VALID_PRIVACY_STATUSES = ('public', 'private', 'unlisted')

logger = logging.getLogger(__name__)

# Authorize the request and store authorization credentials.
def get_authenticated_service():
    flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
    credentials = flow.run_console()
    return build(API_SERVICE_NAME, API_VERSION, credentials = credentials)


def upload_thumbnail(youtube, video_id, file):
    logger.info('Uploading thumbnail...')
    request = youtube.thumbnails().set(
        videoId=video_id,
        media_body=file
    )
    response = request.execute()
    logger.info(response)


def upload_video(youtube, output, metadata):
    logger.info('Starting video upload..')

    video_file = os.path.join(output, 'output.mp4')
    thumbnail_file = os.path.join(output, 'thumbnail.png')

    body = dict(
        snippet=dict(
            title=metadata.title,
            description=metadata.description,
            tags=metadata.tags,
            categoryId=metadata.category_id
        ),

        status=dict(
            privacyStatus=metadata.privacy_status
        )
    )

    insert_request = youtube.videos().insert(
        part=",".join(body.keys()),
        body=body,
        media_body= MediaFileUpload(
            video_file, chunksize=-1, resumable=True)
    )

    video_id = resumable_upload(insert_request)
    upload_thumbnail(youtube, video_id, thumbnail_file)

# This method implements an exponential backoff strategy to resume a
# failed upload.
def resumable_upload(request):
    response = None
    error = None
    retry = 0
    while response is None:
        try:
            logger.debug('Uploading file...')
            status, response = request.next_chunk()
            if response is not None:
                if 'id' in response:
                    print(f'Video id {response['id']} was successfully uploaded.')
                else:
                    exit('The upload failed with an unexpected response: %s' % response)
        except HttpError as e:
            if e.resp.status in RETRIABLE_STATUS_CODES:
                error = 'A retriable HTTP error %d occurred:\n%s' % (e.resp.status,
                                                                    e.content)
            else:
                raise
        except RETRIABLE_EXCEPTIONS as e:
            error = f'A retriable error occurred: {e}'

    if error is not None:
        logger.error(error)
        retry += 1

        if retry > MAX_RETRIES:
            exit('No longer attempting to retry.')

        max_sleep = 2 ** retry
        sleep_seconds = random.random() * max_sleep
        print(f'Sleeping {sleep_seconds} seconds and then retrying...')
        time.sleep(sleep_seconds)