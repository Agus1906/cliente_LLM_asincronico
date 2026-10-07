# Pre-entrega 1 · Unified Async LLM Client

Cliente Python **3.12** con interfaz común para **OpenAI** y **Anthropic**, llamadas
asíncronas, streaming, validación Pydantic y errores controlados.
Toda la documentación de la actividad está en este repositorio.

## 1. Preparar el entorno

En Windows / PowerShell, desde la carpeta del repositorio:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
Copy-Item .env.example .env
```

En Linux / macOS:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
cp .env.example .env
```

No es necesario activar el entorno. Para ejecutar solamente la aplicación,
instalar `requirements.txt`. `requirements-lock.txt`, si está incluido, fija las
versiones efectivamente verificadas y permite reproducir el entorno completo:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

## 2. Configurar `.env`

| Variable | Uso |
| --- | --- |
| `LLM_PROVIDER` | `openai` o `anthropic`; predeterminado `openai` |
| `OPENAI_API_KEY` | Credencial de OpenAI, obligatoria al usar ese proveedor |
| `ANTHROPIC_API_KEY` | Credencial de Anthropic, obligatoria al usar ese proveedor |
| `OPENAI_MODEL` | Identificador del modelo OpenAI habilitado para tu cuenta |
| `ANTHROPIC_MODEL` | Identificador del modelo Anthropic habilitado para tu cuenta |
| `LLM_TEMPERATURE` | Número entre 0 y 2; **Anthropic restringe a 0–1**; predeterminado 0.7 |
| `LLM_MAX_TOKENS` | Entero positivo; predeterminado 256 |
| `LLM_TIMEOUT` | Timeout HTTP del SDK en segundos (0 < valor ≤ 300); predeterminado 30 |
| `LLM_MAX_RETRIES` | Reintentos del SDK de 0 a 5; predeterminado 2 |

Los modelos del ejemplo son configurables: su disponibilidad, límites y soporte
de temperatura dependen del proveedor y de tu cuenta. Para OpenAI se utiliza
Chat Completions y se traduce `max_tokens` a `max_completion_tokens`.
Modelos que rechacen estos parámetros devuelven un error controlado; el cliente
no cambia parámetros silenciosamente.

`.env` está excluido de Git. No publicar claves ni pegarlas en el código.
El entorno del sistema tiene prioridad sobre `.env`; el archivo se carga desde
la carpeta de `main.py`, independientemente del directorio de ejecución.
No hay que configurar ambas claves si se utiliza un solo proveedor.

## 3. Ejecutar la pregunta de validación

```powershell
# Proveedor configurado en .env
.\.venv\Scripts\python.exe main.py
# Elegir uno explícitamente
.\.venv\Scripts\python.exe main.py --provider openai
.\.venv\Scripts\python.exe main.py --provider anthropic
# Probar ambos, aunque uno falle
.\.venv\Scripts\python.exe main.py --provider both
```

Cada proveedor recibe **«¿Qué es la entropía?»** en modo normal y luego streaming.
`both` ejecuta las demostraciones secuencialmente; todas las llamadas de red son
asíncronas. No hace falta paralelizar dos peticiones para que sean no bloqueantes.
Cada demostración realiza dos solicitudes facturables; `both` realiza cuatro,
más los eventuales reintentos. Requiere conexión, claves y saldo/cuota disponibles.
La autenticación de ChatGPT o de GitHub no reemplaza una API key.

Salida: respuesta completa, fragmentos impresos conforme llegan y motivo de fin.
Ante una falla se muestra un error estructurado. El código de salida es 0 si ambas
modalidades funcionan, 1 ante error y 130 por cancelación con Ctrl+C.
Una respuesta terminada por límite de tokens puede estar truncada: revisar
`finish_reason` (`length` en OpenAI, `max_tokens` en Anthropic).

## 4. Arquitectura y uso desde otro programa

```text
main.py                 carga .env y demuestra ambos modos
schemas.py              mensajes, configuración, respuesta, errores y eventos
manager.py              AsyncLLMManager y selección de proveedor
clients/base.py         BaseLLMClient y normalización de errores
clients/providers.py    OpenAIClient / AnthropicClient con SDK oficiales
tests/test_clients.py   pruebas sin acceso a la API
```

```python
from contextlib import aclosing
from manager import AsyncLLMManager
from schemas import ChatMessage

async def consultar():
    messages = [ChatMessage(role="user", content="¿Qué es la entropía?")]
    # En una aplicación propia, cargar .env previamente con load_dotenv().
    async with AsyncLLMManager.from_env() as manager:
        response = await manager.generate(messages)
        if response.ok:
            print(response.content)
        else:
            print(response.error.code, response.error.message)

        async with aclosing(manager.stream(messages)) as events:
            async for event in events:
                if event.type == "text":
                    print(event.text, end="")
                elif event.type == "error":
                    print(event.error.message)
                else:
                    print(event.finish_reason)
```

`generate()` devuelve `ModelResponse`. `stream()` es un **generador asíncrono**
implementado con `async for` y `yield`. Devuelve eventos `StreamEvent`:

- `text`: fragmento en `text` (puede contener parte de una palabra o varios tokens).
- `done`: finalización exitosa con `finish_reason`.
- `error`: falla controlada en `error`; reemplaza al evento `done`.

El consumidor diferencia los errores del texto generado sin analizar cadenas.
Para finalizar antes de consumir todo el stream, usar `aclosing`, como en el
ejemplo: garantiza el cierre inmediato de la conexión incluso con `break`.
Los clientes y el manager son context managers asíncronos; también exponen
`aclose()`. La cancelación de asyncio se propaga para permitir detener tareas.

Los mensajes aceptan solamente `system`, `user` y `assistant`, con texto no vacío.
En Anthropic, los mensajes `system` iniciales se unen en el parámetro separado
`system`; la conversación debe comenzar con `user`. Un `system` después de la
conversación se rechaza, evitando cambiar el sentido de las instrucciones.
No se implementan herramientas, imágenes ni persistencia de conversaciones.

## 5. Errores, reintentos y trazabilidad

| Código | Tratamiento |
| --- | --- |
| `validation` | Entrada inválida; no se envía la solicitud |
| `authentication` | HTTP 401/403; revisar credencial y permisos |
| `rate_limit` | HTTP 429; revisar tasa y cuota |
| `network` | Fallo de conexión, timeout o interrupción de stream |
| `provider` | Otros errores de la API, incluidos 400 y 5xx |

Los SDK realizan sus reintentos acotados con esperas asíncronas para errores
transitorios (conexión, 408, 409, 429 y 5xx). No se añade una segunda capa de
reintentos. `LLM_MAX_RETRIES=2` permite hasta tres intentos por solicitud.
El timeout es del SDK por operación HTTP, **no un plazo global** incluyendo
reintentos. Un 429 puede deberse a cuota agotada: repetirlo no garantiza recuperación.
`retryable` es orientativo para el consumidor, no activa otro reintento.

Si el stream falla después de emitir texto, se conserva el texto parcial y se
emite `error`. **No se reinicia el stream**: repetirlo podría duplicar contenido
y gasto. Las esperas por red utilizan `AsyncOpenAI` y `AsyncAnthropic`.
Los fallos de programación no se ocultan con un `except Exception` general.
Los logs incluyen código de error y estado HTTP; omiten prompts, claves y cuerpos
de respuesta. Pydantic rechaza campos desconocidos y los modelos son inmutables.

## 6. Verificar sin consumir la API

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
```

Las pruebas utilizan los SDK reales con `httpx.MockTransport`; verifican ambos
proveedores, conversión de mensajes, modo completo, streaming SSE, validación,
401/403, 400, 429, 500, fallos de conexión, reintentos, interrupciones con texto
parcial, cierre anticipado, selección por entorno y cancelación sin bloquear
el event loop. No requieren `.env`, saldo, claves reales ni red.

La prueba real con `main.py --provider both` es la comprobación final de la
disponibilidad de los modelos y las credenciales. Las pruebas simuladas no
demuestran conectividad ni permisos de una cuenta real.

### Validación de esta entrega

El 7 de octubre de 2026 se verificó en Windows con **Python 3.12.15**:

- **42 pruebas aprobadas** con pytest.
- `pip check`: sin dependencias incompatibles.
- `main.py --provider both` sin configuración: informa ambos errores, termina
  con código 1 y no muestra un traceback.
- SDKs instalados: OpenAI 2.54.0 y Anthropic 0.125.0; Pydantic 2.13.5.

Las llamadas reales quedan pendientes de configurar API keys y modelos. En el
sandbox de esta sesión, Windows bloqueó el socket interno del event loop de
asyncio; las pruebas se ejecutaron fuera de esa restricción, manteniendo todas
las solicitudes HTTP simuladas. No se alteró la lógica de la aplicación para
evitar esa limitación del entorno.

## 7. Entregar en GitHub

Desde un repositorio local sin remoto, con GitHub CLI autenticado:

```powershell
gh auth login
git add .
git commit -m "Implement unified asynchronous LLM client"
gh repo create unified-async-llm-client --private --source . --remote origin --push
```

Compartir la URL del repositorio con el evaluador y darle acceso si es privado.
No incluir `.env`, `.venv`, cachés ni credenciales. Si ya existe un remoto,
utilizar ese repositorio en lugar de crear otro.

## Fuentes oficiales

- [OpenAI Python API library](https://developers.openai.com/api/reference/python)
- [SDK oficial de Anthropic: async, streaming y retries](https://github.com/anthropics/anthropic-sdk-python)

Se siguen los SDK asíncronos oficiales y sus mecanismos de streaming y reintentos.
