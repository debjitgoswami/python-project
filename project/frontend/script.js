const messageEl = document.getElementById("message");
const charCountEl = document.getElementById("charCount");
const copyBtn = document.getElementById("copyBtn");
const resultEl = document.getElementById("result");
const classifyBtn = document.getElementById("classifyBtn");

messageEl.addEventListener("input", () => {
  const len = messageEl.value.length;
  charCountEl.textContent = `${len} / 1000 characters`;
});

// Clear textarea, result, reset buttons
function clearAll() {
  messageEl.value = "";
  charCountEl.textContent = "0 / 1000 characters";
  resultEl.textContent = "";
  resultEl.classList.remove("visible");
  copyBtn.disabled = true;
}

// Copy classification result text to clipboard
function copyResult() {
  const text = resultEl.textContent;
  if (!text) return;

  navigator.clipboard
    .writeText(text)
    .then(() => {
      copyBtn.textContent = "Copied!";
      setTimeout(() => {
        copyBtn.textContent = "Copy Result";
      }, 1500);
    })
    .catch(() => {
      alert("Failed to copy. Please copy manually.");
    });
}

async function classifyMessage() {
  const message = messageEl.value.trim();

  // Clear previous result & disable copy button
  resultEl.classList.remove("visible");
  resultEl.textContent = "";
  copyBtn.disabled = true;

  if (!message) {
    resultEl.textContent = "⚠️ Please enter a message to classify.";
    resultEl.classList.add("visible");
    return;
  }

  // Disable button and show loading spinner + text
  classifyBtn.disabled = true;
  resultEl.innerHTML = `<span class="spinner"></span>Classifying...`;
  resultEl.classList.add("visible");

  try {
    const response = await fetch("http://127.0.0.1:5000/classify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });

    if (!response.ok) {
      throw new Error(`Server responded with status ${response.status}`);
    }

    const data = await response.json();

    // Show classification result with fade-in
    resultEl.textContent = `Classification: ${data.classification}`;
    copyBtn.disabled = false;
  } catch (error) {
    console.error("Error during classification:", error);
    resultEl.textContent =
      "❌ An error occurred while classifying the message. Please try again.";
  } finally {
    classifyBtn.disabled = false;
  }
}
