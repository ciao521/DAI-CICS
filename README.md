<div align="center">

# 🌐 DAI-CICS
**Institutional Design and Dynamic Analysis of Intervention in the Community-based Integrated Care System**

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg?style=for-the-badge&logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

*A Multi-Agent Simulation Framework with LLM-driven Nudges for Sustainable Healthcare Systems.*
<br>* author :　Yasuhiro IWAI,  * created :　04/10/2026 <br> 
</div>

---

## 🌟 Abstract

Modern society faces a "polycrisis" driven by climate change-induced disasters and a rapidly aging population, with the number of people requiring nursing care projected to reach 9 million in Japan by 2050. Conventional socioeconomic mechanisms—namely, the "Market Mechanism" (exchange) and the "Power Mechanism" (redistribution)—are proving insufficient, often leading to systemic failures and the exploitation of care workers' dedication in local communities.

**DAI-CICS** explores the dynamics of the **"Community Mechanism"** (spontaneous mutual aid). By employing an LLM-based multi-agent simulation , this project quantitatively demonstrates how AI-driven minimal interventions ("nudges") can prevent the collapse of care networks, mitigate burnout, and guide the system toward sustainable *Care Resilience*.

## ✨ Key Features

* 🧠 **Agent-Based Dynamic Analysis:** Models complex social preferences, including altruism, reciprocity, and *Eudaimonia* (virtue-based well-being).
* 🤖 **LLM-Driven Nudging Engine:** Utilizes AI agents to monitor network health and issue "perspective-shifting nudges" that prevent the crowding-out of intrinsic motivation.
* ⚖️ **Normative Economics Integration:** Evaluates systemic sustainability using a modified Social Welfare Function (SWF) that penalizes the exploitation of dedicated workers.
* 📊 **Real-time Tracking:** Seamless integration with Weights & Biases (W&B) for live metric streaming and LLM-as-a-Judge evaluations.

---

## 🔬 Theoretical Framework & Simulation Design

### 🎭 Agent Typology
The simulation is populated by four distinct classes of agents:
1.  **Elder (Patients):** Individuals facing cognitive decline and Social Determinants of Health (SDH) vulnerabilities, at risk of acute hospital dependence.
2.  **Provider / Family (Caregivers):** Driven by both market forces (wages) and community forces (altruism). They possess dynamically updating variables for *fatigue* (burnout risk) and *Eudaimonia*.
3.  **Link Worker / Manager:** Facilitators of social prescribing and facility managers. Managers operate with an *exploitation rate* parameter that can be mitigated via AI nudges.
4.  **AI Watcher:** A systemic overseer that monitors SDH risks and care burdens, executing targeted nudges (N1-N4) to balance the load and foster community connection.

### 🛤️ Experimental Scenarios
* **Scenario A (Homo-economicus):** A baseline model of pure self-interest. The care network fragments, leading to system collapse and runaway social security costs.
* **Scenario B (Over-reliance & Burnout):** Relies heavily on the community mechanism without AI support. Initially functional, it inevitably leads to "altruism exploitation," causing cascading burnout among caregivers.
* **Scenario C (AI-supported Nudging):** The proposed optimized model. AI watchers distribute loads and facilitate social prescribing. Providers accumulate Eudaimonia sustainably, realizing a resilient community care model for 2050.

### 📐 Modified Social Welfare Function (SWF)
To mathematically prove ethical sustainability, we redefine the Social Welfare Function $W$ to penalize systemic exploitation:

$$W=\sum_{i \in Agents} U_i(c_i, l_i) + \lambda \sum_{i \in Providers} E_i - \gamma \sum_{i \in Providers} \max(0, F_i - F_{threshold})$$

Where $U_i$ is traditional economic utility, $E_i$ represents Eudaimonia (virtue-based happiness), $F_i$ is the accumulated fatigue, and $F_{threshold}$ is the burnout threshold.

---

## 🚀 Getting Started

### 1. Prerequisites & Installation
Ensure you have Python 3.9+ installed. Clone the repository and install the required dependencies.

```bash
git clone https://github.com/ciao421/DAI-CICS.git
cd DAI-CICS
pip install -r requirements.txt
````

Set up your environment variables by copying the template:

```bash
cp env-templete.txt .env
```

Populate the `.env` file with your credentials:

  * `AWS_BEARER_TOKEN_BEDROCK`=bedrock-api-key-...
  * `WANDB_API_KEY`=wandb\_v1\_...
  * `ANTHROPIC_API_KEY`=sk-ant-... *(Optional fallback)*

### 2\. Run ABM Experiments

Execute the multi-agent simulation across different scenarios to generate CSV logs and comparative visualizations.

```bash
# Run all scenarios (A, B, C) with 10 seeds and generate plots
python -m src.run_experiment --scenario all --days 100 --seeds 10 --plot

# Run full ablation study (including C-noN2, C-noN3, C-onlyL1)
python -m src.run_experiment --all-including-ablations --days 100 --seeds 10 --plot
```

*Outputs are saved in the `results/` directory (e.g., `comparison_main.png`, `milestones.png`).*

### 3\. LLM Dialogue Simulation (Discharge Coordination)

Simulate a dynamic, multi-disciplinary conference (Care Manager, Doctor, Planner AI) to observe real-time LLM nudging.

```bash
# Live execution via AWS Bedrock / Claude
python -m src.dialogue_sim --tag live
```

### 4\. Weights & Biases Integration

Track experiments and utilize LLM-as-a-Judge for behavioral evaluation.

```bash
# Stream ABM metrics to W&B in real-time
python -m src.wandb_eval --abm-only --scenario C --days 100 --seeds 10

# Run LLM-as-a-Judge evaluation for Scenario C
python -m src.wandb_eval --scenario C --steps 15
```

📊 **View Dashboard:** [W\&B Project: healthcare-collaboration-sim](https://wandb.ai/yshr-i/healthcare-collaboration-sim)

-----

## 📁 Repository Structure

| Module | Description |
| :--- | :--- |
| `src/config.py` | Centralized parameter management (demographics, thresholds, scenarios). |
| `src/agents.py` | Mesa-based agent class definitions (`Elder`, `Provider`, `LinkWorker`, etc.). |
| `src/model.py` | Core ABM logic: daily task generation, state updates, and nudge application. |
| `src/nudges.py` | AI intervention engine defining intervention levels (L1–L4). |
| `src/metrics.py` | Calculation of Milestones, Failure Conditions, and Gini coefficients. |
| `src/plots.py` | Matplotlib-based visualizations for time-series and ablation studies. |
| `src/llm_agents.py`| Prompts and AWS Bedrock API controllers for LLM-driven agents. |

-----

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

-----

## 📬 Author & Contact

**Yasuhiro Iwai**
*Consultant & Data Scientist | AI & Data Strategy*

  * **Portfolio:** [yasuhiroiwai.jp](https://yasuhiroiwai.jp)
  * **Email:** [contact@yasuhiroiwai.jp](mailto:contact@yasuhiroiwai.jp)
  * **GitHub:** [@ciao521](https://www.google.com/search?q=https://github.com/ciao521)
