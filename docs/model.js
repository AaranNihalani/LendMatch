function valueOrDash(value) {
    return value === undefined || value === null ? "--" : String(value);
}

function numberValue(value, digits = 3) {
    const number = Number(value);
    return Number.isFinite(number) ? number.toFixed(digits) : "--";
}

function addDefinition(list, term, value) {
    const dt = document.createElement("dt");
    const dd = document.createElement("dd");
    dt.textContent = term;
    dd.textContent = valueOrDash(value);
    list.append(dt, dd);
}

async function loadModelCard() {
    const response = await fetch("/model-card");
    if (!response.ok) {
        throw new Error("Model card is not available. Train models with python -m src.lendmatch_model.");
    }
    return response.json();
}

function renderModelCard(card) {
    document.getElementById("approvalAuc").textContent = numberValue(card.metrics?.approval?.roc_auc, 4);
    document.getElementById("defaultAuc").textContent = numberValue(card.metrics?.default?.roc_auc, 4);
    document.getElementById("rateMae").textContent = numberValue(card.metrics?.interest_rate?.mae, 3);

    const dataList = document.getElementById("dataSources");
    dataList.innerHTML = "";
    addDefinition(dataList, "Accepted loans", card.data?.accepted_csv);
    addDefinition(dataList, "Rejected applications", card.data?.rejected_csv);
    addDefinition(dataList, "Accepted sample rows", card.data?.accepted_sample_rows?.toLocaleString());
    addDefinition(dataList, "Rejected sample rows", card.data?.rejected_sample_rows?.toLocaleString());

    const notes = document.getElementById("modelNotes");
    notes.innerHTML = "";
    (card.notes || []).forEach((note) => {
        const li = document.createElement("li");
        li.textContent = note;
        notes.appendChild(li);
    });
}

loadModelCard()
    .then(renderModelCard)
    .catch((error) => {
        document.querySelector(".model-page").insertAdjacentHTML(
            "afterbegin",
            `<div class="error-box">${error.message}</div>`,
        );
    });
