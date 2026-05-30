# AstraFarm Irrigation Advisor

**GS - Generative AI for Engineering** · Tema: **Agricultura Inteligente**

Predição da **necessidade de irrigação (litros/m² por mês)** a partir de dados de solo e clima, usando **MLPRegressor** e **XGBRegressor**, com *Feature Selection* e interface em Gradio. O projeto entra como módulo de apoio à plataforma **AstraFarm**.


**Integrantes:**

* Henrique Pontes Oliveira - RM98036
* Pedro Henrique Paladino - RM551180
* Gabriel Diegues Figueiredo Rocha - RM550788

---

## Critérios do GS

| # | Critério | Onde está | Pontos |
|---|----------|-----------|--------|
| 1 | Definição do problema | Notebook §1 | 1,0 |
| 2 | Dataset | §2 | 1,0 |
| 3 | Tratamento e preparação | §3 | 1,0 |
| 4 | Técnicas de treinamento (MLP + XGBoost + Feature Selection) | §4 | 3,0 |
| 5 | Avaliação por métricas (R², MAE, MAPE) | §5 | 1,0 |
| 6 | Avaliação visual | §6 | 1,0 |
| 7 | Código, documentação e deploy | repo + §7 | 2,0 |

---

## Algoritmos utilizados

| Etapa | Algoritmo / Biblioteca | Aula |
|-------|------------------------|------|
| Pré-tratamento | `StandardScaler` | 04 |
| Feature Selection | Correlação (filtro), RFE (wrapper), Importância XGBoost (embedded) | 09 |
| Interpretabilidade | SHAP sobre o XGBoost | 06, 11 |
| Modelo neural | `MLPRegressor`, `hidden_layer_sizes=(64, 32)` | 04, 05 |
| Modelo de boosting | `XGBRegressor`, `n_estimators=1000, max_depth=4, lr=0.1` | 06, 11 |
| Métricas | R², MAPE %, MAE | 04, 11 |
| Deploy | `joblib` + `gradio` | 11 |
| Mapa interativo | `plotly.express` | 04 |

---

## O problema

Em fazendas modernas e em estufas espaciais (AstraFarm, NASA *Veggie*, ESA *EDEN ISS*), a água é um recurso limitado. Irrigar a mais desperdiça; a menos compromete a produção. **Tarefa:** dado o estado do solo (N, P, K, pH) e do clima (temperatura, umidade, chuva), predizer **quantos litros/m² irrigar no próximo ciclo**. É um problema de **regressão tabular**, próximo dos estudos de caso trabalhados na disciplina.

### Como o target foi derivado

O *Crop Recommendation Dataset* é originalmente de classificação. Convertemos para regressão de irrigação aplicando um **balanço hídrico simplificado** (fórmula FAO de evapotranspiração):

```
Irrigação = max(0, ETc − Chuva efetiva)
ETc = 4.5 × temperatura × (1 − umidade/200)
Chuva efetiva = 0.8 × chuva total
```

Em seguida adicionamos **ruído gaussiano** ao target, para simular variações de sensores e condições de campo. Sem isso o alvo seria determinístico.

---

## Dataset

**Crop Recommendation Dataset**: 2.200 registros agrícolas reais (Índia):
- 7 variáveis numéricas: `N, P, K, temperature, humidity, ph, rainfall`
- 1 categórica: `label` (22 culturas)
- Balanceado (100 amostras/cultura), sem dados faltantes
- Baixado do GitHub durante a execução do notebook

---

## Resultados

| Modelo | R² (teste) | MAE (teste) | MAPE % (teste) |
|--------|-----------|-------------|----------------|
| **MLPRegressor** | ~0,99 | ~1,4 L/m² | ~38% |
| XGBRegressor | ~0,98 | ~1,9 L/m² | ~47% |

> O **MAE** foi usado como métrica principal por estar em litros/m². O MAPE fica alto em alguns casos porque o target tem valores pequenos (próximos de 1 L/m²), o que aumenta o percentual relativo do erro.

**Visualizações geradas** (em `outputs/`):
- `01_distribuicao_target.png`: distribuição da irrigação + boxplot por cultura
- `02_correlacao.png`: matriz de correlação (Feature Selection por filtro)
- `03_importancia_features.png`: importância XGBoost (Feature Selection embedded)
- `04_loss_mlp.png`: curva de loss do MLP
- `05_predito_vs_real.png`: dispersão predito × real (ambos modelos)
- `06_residuos.png`: análise de resíduos
- `07_comparativo_modelos.png`: R² e MAPE lado a lado
- `08_shap_importancia.png`: interpretabilidade SHAP do XGBoost
- mapa interativo Plotly (renderiza no notebook)

---

## Como executar

### Instalação
```bash
git clone https://github.com/Pontessxx/Generative-ai-GlobalSoluction-ASTRAFARM
cd gs-astrafarm-irrigacao
pip install -r requirements.txt
```

### Notebook
```bash
jupyter notebook GS_AstraFarm_Irrigacao.ipynb
```
Roda o fluxo completo, incluindo download do dataset, treino, avaliação e geração dos artefatos.

### Deploy (interface web Gradio)
Depois de rodar o notebook ao menos uma vez (gera os artefatos em `outputs/`):
```bash
python app.py
```
Abre a interface no navegador para prever irrigação a partir dos valores informados.

Para gerar um link público temporário do Gradio:
```bash
GRADIO_SHARE=true python app.py
```
No PowerShell:
```powershell
$env:GRADIO_SHARE="true"; python app.py
```

### Google Colab
Faça upload do `.ipynb` e instale as dependências listadas em `requirements.txt`, se necessário.

---

## Estrutura

```
gs-astrafarm-irrigacao/
├── GS_AstraFarm_Irrigacao.ipynb   # Notebook principal (7 critérios)
├── app.py                          # Interface Gradio standalone
├── requirements.txt
├── README.md
├── .gitignore
├── data/                           # dataset usado pelo notebook
└── outputs/                        # (gerado) figuras, modelo, scaler, métricas
```

---

## Integração com a plataforma AstraFarm

| Componente AstraFarm (SECDEVOPS) | Papel neste módulo |
|----------------------------------|--------------------|
| Azure App Service | Hospeda a interface Gradio como API web |
| Azure Key Vault | Credenciais do MLflow / endpoint privado |
| Managed Identity + RBAC | Acesso seguro do app ao modelo serializado |
| GitHub Actions CI/CD | Pipeline de retreino com novos dados |
| Application Insights | Telemetria de predições e *drift* do modelo |

---

## Próximos passos
- Coletar dados reais da estufa via sensores ESP32 (stack já em uso pela equipe) para *fine-tuning*.
- Adicionar estágio fenológico da cultura (germinação/floração/frutificação) como feature, pois afeta diretamente a ETc.
- Testar AutoML com PyCaret como baseline.

---

## Créditos
- **Dataset:** Crop Recommendation Dataset (domínio público, agricultura indiana).
- **Contexto AstraFarm:** baseado no Relatório Técnico SECDEVOPS da equipe AstraFarm.
- Algoritmos e organização seguindo o conteúdo da disciplina *Generative AI for Engineering*.
