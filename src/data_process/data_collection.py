import requests
import hashlib

API_KEY = 'YOUR_API_KEY'
organization = 'correctiv.org'

def generate_claim_id(publisher, review_url):
    raw = f"{publisher}|{review_url}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def get_claim_by_id(claim_id, organization):
    params = {
        'key': YOUR_API_KEY
        'languageCode': 'de',
        'reviewPublisherSiteFilter': organization,
        'pageSize': 10000,
    }

    response = requests.get('https://factchecktools.googleapis.com/v1alpha1/claims:search', params=params)
    data = response.json()

    for claim in data.get('claims', []):
        #claim_text = claim.get('text', '')
        #claimant = claim.get('claimant', '')
        for review in claim.get('claimReview', []):
            review_date =review.get('reviewDate','')
            review_url = review.get('url', '')
            current_id = generate_claim_id(publisher, review_url)

            if current_id == claim_id:
                return {
                    'id': current_id,
                    'claim': claim.get('text'),
                    'reviewUrl': review_url,
                    'rating': review.get('textualRating'),
                    'publisher': review.get('publisher', {}).get('name'),
                    'reviewDate': review.get('reviewDate')
                }

    return None

get_claim_by_id(IDs, publisher)
