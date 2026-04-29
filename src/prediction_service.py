from src.lender_matching import LenderMatcher
from src.lendmatch_model import LendMatchPredictor


class PredictionService:
    def __init__(self, models_dir="models"):
        self.predictor = LendMatchPredictor()
        self.lender_matcher = LenderMatcher()

    def predict(self, input_data):
        result = self.predictor.predict(input_data)
        offers = self.lender_matcher.generate_offers(
            input_data,
            result["predicted_interest_rate"],
            result["default_probability"],
        )
        if result["decision"] == "Needs support":
            offers = []

        result["is_approved"] = result["decision"] == "Eligible"
        result["offers"] = offers
        result["recommendations"] = self._recommendations(input_data, result)
        return result

    def _recommendations(self, input_data, result):
        recommendations = []
        if result["decision"] == "Needs support":
            recommendations.append("Do not route directly to a lender yet; review eligibility barriers with the applicant.")
        if result["default_probability"] >= 0.18:
            recommendations.append("Route to affordability review before lender referral.")
        if float(input_data.get("dti", 0) or 0) > 35:
            recommendations.append("Debt-to-income is high; consider a smaller request or repayment support.")
        if float(input_data.get("fico_score", 0) or 0) < 660:
            recommendations.append("Credit score is below many lender thresholds; offer credit-building guidance.")
        if not recommendations:
            recommendations.append("Applicant profile is suitable for lender matching with standard documentation checks.")
        return recommendations
