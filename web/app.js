const taskInput = document.getElementById("taskInput");
const runTaskBtn = document.getElementById("runTaskBtn");
const taskOutput = document.getElementById("taskOutput");
const ingestPath = document.getElementById("ingestPath");
const ingestBtn = document.getElementById("ingestBtn");
const ingestOutput = document.getElementById("ingestOutput");
const kanBtn = document.getElementById("kanBtn");
const kanOutput = document.getElementById("kanOutput");
const refreshRunsBtn = document.getElementById("refreshRunsBtn");
const runList = document.getElementById("runList");

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`${response.status} ${response.statusText}: ${text}`);
  }
  return response.json();
}

function print(outputElement, payload) {
  outputElement.textContent = typeof payload === "string" ? payload : JSON.stringify(payload, null, 2);
}

async function loadRuns() {
  try {
    const data = await fetchJson("/runs");
    runList.innerHTML = "";
    data.runs.forEach((runName) => {
      const li = document.createElement("li");
      li.textContent = runName;
      li.addEventListener("click", async () => {
        const run = await fetchJson(`/runs/${runName}`);
        print(taskOutput, run);
      });
      runList.appendChild(li);
    });
  } catch (error) {
    print(taskOutput, error.message);
  }
}

runTaskBtn.addEventListener("click", async () => {
  taskOutput.textContent = "Running task...";
  try {
    const result = await fetchJson("/run-task", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task: taskInput.value.trim() }),
    });
    print(taskOutput, result);
    await loadRuns();
  } catch (error) {
    print(taskOutput, error.message);
  }
});

ingestBtn.addEventListener("click", async () => {
  ingestOutput.textContent = "Ingesting...";
  try {
    const result = await fetchJson("/ingest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: ingestPath.value.trim() }),
    });
    print(ingestOutput, result);
  } catch (error) {
    print(ingestOutput, error.message);
  }
});

kanBtn.addEventListener("click", async () => {
  kanOutput.textContent = "Running KAN demo...";
  try {
    const result = await fetchJson("/kan-demo");
    print(kanOutput, result);
  } catch (error) {
    print(kanOutput, error.message);
  }
});

refreshRunsBtn.addEventListener("click", loadRuns);
window.addEventListener("load", loadRuns);
