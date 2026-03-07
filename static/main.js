document.getElementById('loanForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const submitBtn = document.getElementById('submitBtn');
    const originalBtnText = submitBtn.innerHTML;
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<span class="flex items-center justify-center"><svg class="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg> Processing...</span>';

    // Get form data
    const formData = new FormData(e.target);
    const data = {
        loan_amount: parseFloat(formData.get('loan_amount')),
        annual_inc: parseFloat(formData.get('annual_inc')),
        fico_score: parseFloat(formData.get('fico_score')),
        dti: parseFloat(formData.get('dti')),
        state: formData.get('state'),
        term: parseInt(formData.get('term')),
        emp_length: formData.get('emp_length'),
        home_ownership: formData.get('home_ownership'),
        purpose: 'debt_consolidation', 
        revol_bal: 10000.0, 
        total_acc: 20.0 
    };

    // UI Transitions
    const initialState = document.getElementById('initialState');
    const loadingState = document.getElementById('loadingState');
    const resultsSection = document.getElementById('resultsSection');

    if(initialState) initialState.classList.add('hidden');
    if(resultsSection) resultsSection.classList.add('hidden');
    if(loadingState) loadingState.classList.remove('hidden');

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });

        if (!response.ok) throw new Error('Prediction failed');

        const result = await response.json();
        
        // Wait a small delay to show loading animation (UX)
        setTimeout(() => {
            displayResults(result);
            if(loadingState) loadingState.classList.add('hidden');
            if(resultsSection) resultsSection.classList.remove('hidden');
        }, 600);
        
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred. Please check console.');
        if(loadingState) loadingState.classList.add('hidden');
        if(initialState) initialState.classList.remove('hidden');
    } finally {
        submitBtn.disabled = false;
        submitBtn.innerHTML = originalBtnText;
    }
});

function displayResults(result) {
    const approvalElem = document.getElementById('approvalResult');
    const approvalCard = document.getElementById('approvalCard');
    const approvalProb = document.getElementById('approvalProb');
    const rateElem = document.getElementById('rateResult');
    const riskElem = document.getElementById('riskResult');
    const offersContainer = document.getElementById('lenderOffers');
    
    // Approval Logic
    const probPercent = (result.approval_probability * 100).toFixed(1);
    
    if (result.is_approved) {
        approvalElem.textContent = "Approved";
        approvalElem.className = "text-2xl font-bold text-green-600";
        approvalProb.textContent = `${probPercent}% Probability`;
        approvalProb.className = "text-sm font-medium mt-1 text-green-700";
        approvalCard.className = "bg-green-50 p-5 rounded-xl shadow-sm border border-green-200 flex flex-col justify-between";
    } else {
        approvalElem.textContent = "Rejected";
        approvalElem.className = "text-2xl font-bold text-red-600";
        approvalProb.textContent = `${probPercent}% Probability`;
        approvalProb.className = "text-sm font-medium mt-1 text-red-700";
        approvalCard.className = "bg-red-50 p-5 rounded-xl shadow-sm border border-red-200 flex flex-col justify-between";
    }
    
    // Interest Rate
    if (result.predicted_interest_rate) {
        rateElem.textContent = `${result.predicted_interest_rate.toFixed(2)}%`;
    } else {
        rateElem.textContent = "N/A";
    }
    
    // Default Risk
    if (result.default_probability) {
        riskElem.textContent = `${(result.default_probability * 100).toFixed(1)}%`;
    } else {
        riskElem.textContent = "N/A";
    }
    
    // Offers
    offersContainer.innerHTML = '';
    if (result.offers && result.offers.length > 0) {
        result.offers.forEach((offer, index) => {
            const div = document.createElement('div');
            div.className = 'p-4 hover:bg-slate-50 transition-colors flex justify-between items-center group';
            div.innerHTML = `
                <div class="flex items-center gap-4">
                    <div class="w-10 h-10 rounded-full bg-blue-100 flex items-center justify-center text-blue-600 font-bold text-sm">
                        ${offer.lender_name.substring(0,2).toUpperCase()}
                    </div>
                    <div>
                        <h4 class="font-semibold text-slate-900 group-hover:text-blue-700 transition-colors">${offer.lender_name}</h4>
                        <div class="text-sm text-slate-500">$${offer.monthly_payment}/mo • ${offer.term} months</div>
                    </div>
                </div>
                <div class="text-right">
                    <div class="text-lg font-bold text-slate-900">${offer.interest_rate}%</div>
                    <div class="text-xs text-slate-500">APR</div>
                </div>
            `;
            offersContainer.appendChild(div);
        });
    } else {
        offersContainer.innerHTML = `
            <div class="p-8 text-center">
                <p class="text-slate-500 text-sm">No matching offers found based on your criteria.</p>
            </div>
        `;
    }
}