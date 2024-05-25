# Tutorial de RAG con langchain y chroma vector db

Primero se instalan las dependencias que tenemos en el archivo .txt

```python
pip install -r requirements.txt
```

Despues creamos la base de datos de Chroma

```python
python create_database.py
```

Despues corremos la query que gustemos de los archivos cargados en el LLM

```python
python query_data.py "What is the main theme of 'The Deadline'?"
```
