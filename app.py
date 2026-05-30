"""
Interface Gradio do AstraFarm Irrigation Advisor.

Carrega o modelo treinado (joblib) e expõe uma interface Gradio para
prever a necessidade de irrigação (L/m²).

Uso:
    1) Rode o notebook GS_AstraFarm_Irrigacao.ipynb uma vez para gerar
       os arquivos em outputs/ (modelo, scaler, colunas).
    2) python app.py
"""
import joblib
import os
import pandas as pd
import gradio as gr

# Artefatos gerados pelo notebook
modelo = joblib.load("outputs/modelo_irrigacao.joblib")
scaler = joblib.load("outputs/scaler_irrigacao.joblib")
colunas_modelo = joblib.load("outputs/colunas_irrigacao.joblib")

numerical_columns = ["N", "P", "K", "temperature", "humidity", "ph",
                     "rainfall"]

CULTURAS = sorted(c.replace("crop_", "") for c in colunas_modelo
                  if c.startswith("crop_"))


def prever_irrigacao(N, P, K, temperatura, umidade, ph, chuva_mensal,
                    cultura):
    entrada = pd.DataFrame([{
        "N": N, "P": P, "K": K,
        "temperature": temperatura, "humidity": umidade,
        "ph": ph, "rainfall": chuva_mensal
    }])
    for col in colunas_modelo:
        if col.startswith("crop_"):
            entrada[col] = 1 if col == f"crop_{cultura}" else 0
    entrada = entrada[colunas_modelo]
    entrada[numerical_columns] = scaler.transform(entrada[numerical_columns])

    previsao = modelo.predict(entrada)[0]
    return f"Irrigação recomendada: {previsao:.2f} L/m² no mês"


with gr.Blocks(title="AstraFarm Irrigation Advisor") as app:
    gr.Markdown(
        "# AstraFarm Irrigation Advisor\n\n"
        "Predição da necessidade de irrigação (L/m² por mês) a partir de "
        "dados de solo e clima.")

    with gr.Row():
        with gr.Column():
            N = gr.Number(label="Nitrogênio (kg/ha)", value=90)
            P = gr.Number(label="Fósforo (kg/ha)", value=42)
            K = gr.Number(label="Potássio (kg/ha)", value=43)
        with gr.Column():
            temperatura = gr.Number(label="Temperatura média (°C)",
                                    value=25.0)
            umidade = gr.Number(label="Umidade relativa (%)", value=70.0)
            ph = gr.Number(label="pH do solo", value=6.5)
        with gr.Column():
            chuva = gr.Number(label="Chuva no mês (mm)", value=80.0)
            cultura = gr.Dropdown(label="Cultura", choices=CULTURAS,
                                  value=CULTURAS[0] if CULTURAS else None)

    saida = gr.Textbox(label="Resultado", lines=2)

    with gr.Row():
        btn = gr.Button("Prever irrigação", variant="primary")
        btn_clear = gr.Button("Limpar", variant="secondary")

    btn.click(fn=prever_irrigacao,
              inputs=[N, P, K, temperatura, umidade, ph, chuva, cultura],
              outputs=saida)
    btn_clear.click(
        fn=lambda: [90, 42, 43, 25.0, 70.0, 6.5, 80.0,
                    CULTURAS[0] if CULTURAS else None, ""],
        inputs=[],
        outputs=[N, P, K, temperatura, umidade, ph, chuva, cultura, saida])


if __name__ == "__main__":
    share_publico = os.getenv("GRADIO_SHARE", "false").lower() == "true"
    app.launch(share=share_publico)
