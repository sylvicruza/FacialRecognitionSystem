import requests

def fetch_user_data(last_name):
    # API endpoint for searching by last name
    url = f"https://openedheavenschapel.co.uk/churchcrm/api/persons/search/{last_name}"

    # API key for authorization
    headers = {
        'x-api-key': 'EtqByhhjSGRS8cv6jxO4rMHvJOadtMlVrDnNDXSv3OfY7eH1m3',
    }

    # Make the GET request
    response = requests.get(url, headers=headers)

    # Check if the response is successful
    if response.status_code == 200:
        return response.json()  # Return the JSON response as a list of users
    else:
        return None  # If there is an error, return None


def fetch_user_data_by_id(portal_id):
    # API endpoint for fetching user data by portal_id
    url = f"https://openedheavenschapel.co.uk/churchcrm/api/person/{portal_id}"

    # API key for authorization
    headers = {
        'x-api-key': 'EtqByhhjSGRS8cv6jxO4rMHvJOadtMlVrDnNDXSv3OfY7eH1m3',
    }

    # Make the GET request
    response = requests.get(url, headers=headers)

    # Check if the response is successful
    if response.status_code == 200:
        return response.json()
    else:
        return None  # Return None if no data is found or the request fails
