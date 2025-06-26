import requests
import time

def get_location_nominatim(address):
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        'q': address,
        'format': 'json',
        'addressdetails': 1,
        'limit': 1,
        'accept-language': 'en'
    }
    headers = {
        'User-Agent': 'YourAppName/1.0 (your.email@example.com)'  # Replace with your app name/email
    }

    response = requests.get(url, params=params, headers=headers)
    data = response.json()

    if not data:
        return None, None, None  # No results

    address_details = data[0]['address']
    city = address_details.get('city') or address_details.get('town') or address_details.get('village')
    province = address_details.get('state') or address_details.get('region')
    country = address_details.get('country')

    return {'city': city, 'province': province, 'country': country}

# Example usage
if __name__ == "__main__":
    address = "160, Sapyeong-daero"
    city, province, country = get_location_nominatim(address)
    print(f"City: {city}\nProvince/State: {province}\nCountry: {country}")

