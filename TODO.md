- Mejorar funcion de rewards
- Entrenar red Q en entorno simulado con 1000 episodios sin render y guardar pesos:
   `uv run python src/doro/utils/train_agent.py --episodes 1000`
- Entrenar LSTM de nuevo con el nuevo modulo `LSTMFeatureExtractor`
- Rellenar execute_action usando control.Control: (main.py del cliente libreria Freenove)
  