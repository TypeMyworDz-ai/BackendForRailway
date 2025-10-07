# diagnose_firebase.py
import os
import base64
import json
import sys

config_base64 = os.environ.get('FIREBASE_ADMIN_SDK_CONFIG_BASE64')

if config_base64:
    try:
        decoded_json = base64.b64decode(config_base64).decode('utf-8')
        parsed_json = json.loads(decoded_json)
        print('Firebase config decoded and parsed successfully.')
        print('Project ID: {}'.format(parsed_json.get('project_id', 'N/A')))
        print('Client Email: {}'.format(parsed_json.get('client_email', 'N/A')))
        # Optionally print more, but keep it brief for security
        # print('Full decoded JSON (WARNING - sensitive!): {}'.format(decoded_json))
    except Exception as e:
        print('ERROR: Failed to decode/parse Firebase config: {}'.format(e))
        sys.exit(1) # Exit with error code
else:
    print('FIREBASE_ADMIN_SDK_CONFIG_BASE64 environment variable is NOT set in Dockerfile context.')
    sys.exit(1) # Exit with error code

