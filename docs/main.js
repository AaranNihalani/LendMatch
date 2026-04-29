const form = document.getElementById("loanForm");
const submitBtn = document.getElementById("submitBtn");
const emptyState = document.getElementById("emptyState");
const resultsPanel = document.getElementById("resultsPanel");
const healthPanel = document.getElementById("healthPanel");
const appError = document.getElementById("appError");

const examples = [
    {
        loan_amount: 15000,
        annual_inc: 90000,
        fico_score: 800,
        dti: 10,
        state: "CA",
        term: 36,
        emp_length: "5 years",
        home_ownership: "MORTGAGE",
        purpose: "debt_consolidation",
        credit_history_years: 15,
        revol_util: 20,
        open_acc: 12,
    },
    {
        loan_amount: 8500,
        annual_inc: 32000,
        fico_score: 632,
        dti: 39.2,
        state: "TX",
        term: 36,
        emp_length: "2 years",
        home_ownership: "RENT",
        purpose: "medical",
        credit_history_years: 5,
        revol_util: 71,
        open_acc: 6,
    },
];

let exampleIndex = 0;

function asPercent(value) {
    return `${(Number(value || 0) * 100).toFixed(1)}%`;
}

function money(value) {
    return Number(value || 0).toLocaleString("en-US", {
        style: "currency",
        currency: "USD",
        maximumFractionDigits: 0,
    });
}

function getPayload() {
    const data = new FormData(form);
    const numericFields = [
        "loan_amount",
        "annual_inc",
        "fico_score",
        "dti",
        "term",
        "credit_history_years",
        "revol_util",
        "open_acc",
    ];
    const payload = {};
    for (const [key, value] of data.entries()) {
        payload[key] = numericFields.includes(key) ? Number(value) : value;
    }
    payload.revol_bal = Math.round(payload.annual_inc * 0.15);
    payload.total_acc = Math.max(payload.open_acc + 8, 12);
    payload.verification_status = "Source Verified";
    payload.application_type = "Individual";
    return payload;
}

function setDecisionStyle(decision) {
    const strip = document.getElementById("decisionStrip");
    strip.style.borderColor = "#d9e2e8";
    strip.style.background = "#fff";
    if (decision === "Eligible") {
        strip.style.borderColor = "#a8dcc6";
        strip.style.background = "#f0faf5";
    }
    if (decision === "Needs support") {
        strip.style.borderColor = "#ffc9c2";
        strip.style.background = "#fff7f5";
    }
}

function renderOffers(offers) {
    const container = document.getElementById("offersList");
    const count = document.getElementById("offerCount");
    container.innerHTML = "";
    count.textContent = `${offers.length} ${offers.length === 1 ? "match" : "matches"}`;

    if (!offers.length) {
        container.innerHTML = `<div class="error-box">No partner offers matched this profile. Use the guidance below before referral.</div>`;
        return;
    }

    offers.forEach((offer) => {
        const row = document.createElement("article");
        row.className = "offer";
        row.innerHTML = `
            <div>
                <strong>${offer.lender_name}</strong>
                <span>${money(offer.monthly_payment)}/month over ${offer.term} months · confidence ${offer.confidence_score}%</span>
            </div>
            <div class="offer-rate">${Number(offer.interest_rate).toFixed(2)}%</div>
        `;
        container.appendChild(row);
    });
}

function renderRecommendations(items) {
    const list = document.getElementById("recommendations");
    list.innerHTML = "";
    items.forEach((item) => {
        const li = document.createElement("li");
        li.textContent = item;
        list.appendChild(li);
    });
}

function renderResults(result) {
    appError.classList.add("hidden");
    emptyState.classList.add("hidden");
    resultsPanel.classList.remove("hidden");

    document.getElementById("decisionText").textContent = result.decision;
    document.getElementById("riskBand").textContent = `${result.risk_band} risk`;
    document.getElementById("approvalMetric").textContent = asPercent(result.approval_probability);
    document.getElementById("defaultMetric").textContent = asPercent(result.default_probability);
    document.getElementById("rateMetric").textContent = `${Number(result.predicted_interest_rate).toFixed(2)}%`;
    document.getElementById("approvalBar").style.width = asPercent(result.approval_probability);
    document.getElementById("defaultBar").style.width = asPercent(result.default_probability);

    setDecisionStyle(result.decision);
    renderOffers(result.offers || []);
    renderRecommendations(result.recommendations || []);
}

async function checkHealth() {
    try {
        const response = await fetch("/health");
        const health = await response.json();
        const dot = healthPanel.querySelector(".status-dot");
        const text = healthPanel.querySelector("p");
        if (health.models_loaded) {
            dot.classList.add("ready");
            text.textContent = "Trained models loaded and ready";
        } else {
            dot.classList.add("error");
            text.textContent = "Train models with python -m src.lendmatch_model";
        }
    } catch {
        healthPanel.querySelector(".status-dot").classList.add("error");
        healthPanel.querySelector("p").textContent = "API is not reachable";
    }
}

document.getElementById("loadExample").addEventListener("click", () => {
    exampleIndex = (exampleIndex + 1) % examples.length;
    const example = examples[exampleIndex];
    Object.entries(example).forEach(([key, value]) => {
        const input = form.elements[key];
        if (input) input.value = value;
    });
});

form.addEventListener("submit", async (event) => {
    event.preventDefault();
    submitBtn.disabled = true;
    submitBtn.textContent = "Analyzing...";

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(getPayload()),
        });
        const payload = await response.json();
        if (!response.ok) {
            throw new Error(payload.detail || "Prediction request failed");
        }
        renderResults(payload);
    } catch (error) {
        appError.textContent = error.message;
        appError.classList.remove("hidden");
        emptyState.classList.remove("hidden");
        resultsPanel.classList.add("hidden");
    } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = "Analyze application";
    }
});

checkHealth();
