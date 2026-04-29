import pandas as pd

class LenderMatcher:
    def __init__(self):
        # Simulated Lender Database
        # In a real system, this would come from a database
        self.lenders = [
            {
                "id": "L001",
                "name": "Community Prime Bank",
                "min_score": 720,
                "max_dti": 35.0,
                "min_income": 60000,
                "max_loan": 50000,
                "states": ["CA", "NY", "TX", "FL", "IL"],
                "base_rate_adj": -0.5 # Competitive rates
            },
            {
                "id": "L002",
                "name": "Credit Union Plus",
                "min_score": 660,
                "max_dti": 40.0,
                "min_income": 40000,
                "max_loan": 35000,
                "states": "ALL",
                "base_rate_adj": 0.0
            },
            {
                "id": "L003",
                "name": "Starter Access Fund",
                "min_score": 600,
                "max_dti": 50.0,
                "min_income": 30000,
                "max_loan": 20000,
                "states": "ALL",
                "base_rate_adj": 2.5 # Higher risk, higher rate
            },
            {
                "id": "L004",
                "name": "Responsible Lending Network",
                "min_score": 640,
                "max_dti": 45.0,
                "min_income": 35000,
                "max_loan": 40000,
                "states": "ALL",
                "base_rate_adj": 1.0
            },
            {
                "id": "L005",
                "name": "Impact Finance Partner",
                "min_score": 680,
                "max_dti": 40.0,
                "min_income": 50000,
                "max_loan": 45000,
                "states": ["CA", "WA", "MA", "TX"],
                "base_rate_adj": -0.2
            }
        ]

    def match_lenders(self, application_data):
        """
        Match an application to suitable lenders based on hard constraints.
        
        Args:
            application_data (dict): Dictionary containing:
                - fico_score (float)
                - dti (float)
                - annual_inc (float)
                - loan_amount (float)
                - state (str)
                
        Returns:
            list: List of matching lender dictionaries.
        """
        matches = []
        
        fico = application_data.get('fico_score', 0)
        dti = application_data.get('dti', 100)
        income = application_data.get('annual_inc', 0)
        amount = application_data.get('loan_amount', 0)
        state = application_data.get('state', 'Unknown')
        
        for lender in self.lenders:
            # Check FICO
            if fico < lender['min_score']:
                continue
                
            # Check DTI
            if dti > lender['max_dti']:
                continue
                
            # Check Income
            if income < lender['min_income']:
                continue
                
            # Check Loan Amount
            if amount > lender['max_loan']:
                continue
                
            # Check State
            if lender['states'] != "ALL" and state not in lender['states']:
                continue
                
            matches.append(lender)
            
        return matches

    def generate_offers(self, application_data, predicted_base_rate, predicted_default_prob):
        """
        Generate specific offers from matched lenders.
        
        Args:
            application_data (dict): Application details.
            predicted_base_rate (float): The ML-predicted fair interest rate.
            predicted_default_prob (float): The ML-predicted default probability.
            
        Returns:
            list: List of offer dictionaries.
        """
        matched_lenders = self.match_lenders(application_data)
        offers = []
        
        for lender in matched_lenders:
            # Adjust rate based on lender strategy
            # Lenders might charge more if default prob is higher, but here we simplify
            # using the base_rate_adj and the ML predicted rate.
            
            # If default prob is very high (> 20%), even matched lenders might decline or hike rate
            risk_premium = 0
            if predicted_default_prob > 0.10:
                risk_premium = 2.0
            if predicted_default_prob > 0.20:
                risk_premium = 5.0
                
            final_rate = predicted_base_rate + lender['base_rate_adj'] + risk_premium
            
            # Cap rate at reasonable limits
            final_rate = max(5.0, min(30.0, final_rate))
            
            offer = {
                "lender_id": lender['id'],
                "lender_name": lender['name'],
                "interest_rate": round(final_rate, 2),
                "monthly_payment": self._calculate_payment(
                    application_data.get('loan_amount'), 
                    final_rate, 
                    int(application_data.get("term", 36) or 36)
                ),
                "term": int(application_data.get("term", 36) or 36),
                "confidence_score": round((1 - predicted_default_prob) * 100, 1)
            }
            offers.append(offer)
            
        return offers

    def _calculate_payment(self, principal, rate, term_months):
        if rate == 0:
            return principal / term_months
        
        r = rate / 100 / 12
        payment = principal * (r * (1 + r)**term_months) / ((1 + r)**term_months - 1)
        return round(payment, 2)
