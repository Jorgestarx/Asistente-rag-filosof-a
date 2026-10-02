import ollama
import math

#-----------------LECTURA Y CHUNKING----------------------------

# "r" me dice el modo en que se abre el archivo, en este caso es r de read.
#encoding lo usamos para determinar el tipo de enconding correcto que acepte tildes y la ñ, de caso contrario python puede aumir uno incorrecto y no leer las tildes.
with open("apuntes_filosofia.txt", "r", encoding="utf-8") as archivo:
    contenido = archivo.read()

parrafos = contenido.split("\n\n")

(len(parrafos))

#Lista donde se guardan los chunks correjidos en cuanto a lo títulos
chunks_finales = []

i = 0

#este while se usa para evaluar lo párrafos y títulos, de esta forma unimos los títulos con su párrafo respectivo, y que la IA pueda tener contexto de lo que lee
while i <len(parrafos):
    if len(parrafos[i]) <100:
        nuevo_parrafo = (f"{parrafos[i]} \n {parrafos[i+1]}")
        chunks_finales.append(nuevo_parrafo)
        i = i+2
    else:
        chunks_finales.append(parrafos[i])
        i = i+1

print(f"chunks finales = {len(chunks_finales)}")


#----------------------Convertir Chunks en vectores--------------------------
chunks_embedd = []

for chunk in chunks_finales:
    resultado = ollama.embeddings(model="nomic-embed-text", prompt=chunk)
    chunk_con_embedding = {"texto": chunk, "embedding": resultado["embedding"]}
    chunks_embedd.append(chunk_con_embedding)


#--------------Cálcular producto punto vectores chunks-----------------
#zip recorre las dos listas vector1 y vector2 al mismo tiempo | v1 * v2 me cálcula una parte del producto punto, la termino de calcular con la suma de todas las multiplicaciones
def similitudpp(vector1, vector2):
    producto_punto = []
    for v1, v2 in zip(vector1,vector2):
        producto_puntovector = v1 * v2
        producto_punto.append(producto_puntovector)
    return sum(producto_punto)


#Calcular magnitud vectores
def similitudmag (vector):
    magnitud = []
    for valor in vector:
        calculo_magnitud = (valor**2)
        magnitud.append(calculo_magnitud)
    return math.sqrt(sum(magnitud))


#Juntamos las funciones de producto punto y magnitud para calcular la similitud coseno
def similitud_coseno(vector1, vector2):
    producto = similitudpp(vector1, vector2)
    magnitud1 = similitudmag(vector1)
    magnitud2 = similitudmag(vector2)
    return producto / (magnitud1 * magnitud2)


#Convertimos en vector la pregunta del usuario para poder compararla con los chunks
pregunta = input("Pregunta lo que quieras sobre el mito de la caverna: ")

pregunta_embedding = ollama.embeddings(model="nomic-embed-text", prompt=pregunta)

#Generamos variable para ir guardando la mejor similitud con la pregunta del usuario y el chunk que le corresponde
resultados = []

for chunk in chunks_embedd:
    similitud = similitud_coseno(chunk["embedding"], pregunta_embedding["embedding"])
    cálculo = {"texto": chunk["texto"], "similitud": similitud}
    resultados.append(cálculo)

resultados_ordenados = sorted(resultados, key=lambda d:d["similitud"], reverse=True)

#escogemos solo los 3 chunks con mayor similitud
top_3 = resultados_ordenados[:3]

#en contexto inicial solo se usa el contenido del texto, no la similitud.
contexto_texto = "\n\n".join(item["texto"] for item in top_3)
contexto = f"Contexto : {contexto_texto} \n\n Pregunta: {pregunta}"

respuesta = ollama.chat(model="llama3.2", messages=[{"role": "system", "content": "Eres un experto en filosofía, y principalmente sobre el mito de la caverna, reponde utilizando el contexto que se te da"},
                                                    {"role": "user", "content": contexto}])

print(respuesta["message"]["content"])