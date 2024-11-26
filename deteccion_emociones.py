import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import nltk

data = pd.DataFrame({
    'Texto': [
        'Estoy muy feliz con esta noticia', 'No puedo creer que haya fallado',
        'Me frustra demasiado esta situación', 'Qué alegría verte después de tanto tiempo',
        'Estoy molesto con lo que hiciste', 'Esto me hace sentir muy triste',
        'Estoy satisfecho con mi desempeño', 'Me siento muy mal por lo que pasó',
        'Esto es una completa pérdida de tiempo', 'Hoy fue un gran día para mí',
        'Estoy emocionado de verte', 'Me siento muy frustrado por esta situación',
        'Qué bueno verte sonreír', 'No puedo soportar más esta presión',
        'Estoy extremadamente feliz con los resultados', 'Es completamente frustrante lo que pasó',
        'Me duele mucho esto, me siento devastado', 'Hoy fue un día espectacular',
        'Esto es una injusticia total'
    ],
    'Emoción': ['Alegría', 'Tristeza', 'Ira', 'Alegría', 'Ira',
                'Tristeza', 'Alegría', 'Tristeza', 'Ira', 'Alegría',
                'Alegría', 'Ira', 'Alegría', 'Tristeza', 'Alegría',
                'Ira', 'Tristeza', 'Alegría', 'Ira']
})

def aumentar_datos(texto, emocion):
    sinonimos = {
        'feliz': ['contento', 'alegre', 'encantado'],
        'triste': ['desanimado', 'abatido', 'afligido'],
        'frustrado': ['irritado', 'desesperado', 'molesto']
    }
    palabras = texto.split()
    nuevas_frases = []
    for palabra in palabras:
        if palabra in sinonimos:
            for sinonimo in sinonimos[palabra]:
                nueva_frase = texto.replace(palabra, sinonimo)
                nuevas_frases.append((nueva_frase, emocion))
    return nuevas_frases

for i, row in data.iterrows():
    data = pd.concat([data, pd.DataFrame(aumentar_datos(row['Texto'], row['Emoción']),
    columns=['Texto', 'Emoción'])], ignore_index=True)

stop_words = set(stopwords.words('spanish'))
lemmatizer = WordNetLemmatizer()

def limpiar_texto(texto):
    texto = texto.lower()
    texto = texto.translate(str.maketrans('', '', string.punctuation))
    palabras = word_tokenize(texto, language='spanish')
    palabras_limpiadas = [lemmatizer.lemmatize(palabra) for palabra in palabras if palabra not in stop_words]
    return ' '.join(palabras_limpiadas)

data['Texto_Limpio'] = data['Texto'].apply(limpiar_texto)

vectorizador = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
X = vectorizador.fit_transform(data['Texto_Limpio'])
y = data['Emoción']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'class_weight': ['balanced']
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X_train, y_train)

modelo = grid_search.best_estimator_

y_pred = modelo.predict(X_test)

print("Mejores Hiperparámetros:", grid_search.best_params_)
print("Reporte de Clasificación:\n", classification_report(y_test, y_pred))
print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred))

def predecir_emocion(texto):
    texto_limpio = limpiar_texto(texto)
    vector = vectorizador.transform([texto_limpio])
    return modelo.predict(vector)[0]

nueva_frase = "Estoy muy emocionado por lo que logré"
print(f"Emoción detectada para '{nueva_frase}': {predecir_emocion(nueva_frase)}")

nueva_frase = "No puedo soportar más esta situación"
print(f"Emoción detectada para '{nueva_frase}': {predecir_emocion(nueva_frase)}")

nueva_frase = "Me siento triste, devastado, nada tiene sentido"
print(f"Emoción detectada para '{nueva_frase}': {predecir_emocion(nueva_frase)}")
