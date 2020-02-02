# TWilBert: Pre-trained Deep Bidirectional Transformers for Spanish Twitter
***

![](https://i.imgur.com/tYZq8Tj.png)


**TWilBert** es una adaptación del modelo [**BERT**](https://arxiv.org/abs/1802.05365) con modificaciones publicadas en diversos papers ([**ALBERT**](https://arxiv.org/abs/1909.11942), [**Product Key Memories**](https://arxiv.org/abs/1907.05242) o [**RoBERTa**](https://arxiv.org/abs/1907.11692)), para el aprendizaje de embeddings contextualizados. El objetivo es impulsar el estado del arte en tareas de Procesamiento del Lenguaje Natural en **español** dedicadas a la red social **Twitter**. Por ello, proporcionamos modelos pre-entrenados especializados en el idioma español y el dominio de Twitter. Así, se pretende establecer un nuevo y competitivo baseline, que permita que los investigadores focalicen su atención en el desarrollo de nuevas y mejores arquitecturas para tareas NLP en Twitter español.

**TWilBert** se proporciona como un framework que permite el pre-entrenamiento, la evaluación y el finetuning de modelos BERT-like, con el que se han obtenido resultados estado del arte para gran cantidad de tareas propuestas en diversos congresos internacionales sobre clasificación de textos españoles en Twitter. Si usas este código y los modelos pre-entrenados, por favor, cita la siguiente referencia:
~~~
@article{TWilBert,
  title={TWilBert: Pre-trained Deep Bidirectional Transformers for Spanish Twitter},
  author={Gonz{\'a}lez, Jos{\'e}-{\'A}ngel and Hurtado, Llu’{\i}s-F and Pla, Ferran},
  journal={To be decided},
  year={Forthcoming}
}
~~~

# Características
***
**TWilBert** es un entorno que permite abordar cualquier tipo de tarea de NLP aplicada a redes sociales, basándose en modelos de embeddings contextualizados BERT-like, incluyendo gran parte de las herramientas publicadas en papers como [**BERT**](https://arxiv.org/abs/1802.05365) con modificaciones publicadas en diversos papers ([**ALBERT**](https://arxiv.org/abs/1909.11942), [**Product Key Memories**](https://arxiv.org/abs/1907.05242) o [**RoBERTa**](https://arxiv.org/abs/1907.11692). Se proporcionan implementaciones de:

* **Transformer Encoders**: Self-Attention, Multi-Head Self-Attention, Product Key Memories y técnicas como Cross-Layer Parameter Sharing o Factorized Embedding Parameterization.
* **Optimizadores**: Gradient Accumulation, Adam, LAMB y Learning Rate Schedules (NOAM)
* **Modelos**: creación de modelos TWilBert genéricos y modelos de fine-tuning integrados en el entorno.
* **Aplicaciones**: aplicaciones para entrenar, evaluar y hacer finetuning de modelos TWilBert.
* **Utilidades**: generación y preparación de muestras para modelos BERT-like, preprocesado y creación de vocabularios subword. 
* **Monitorización**: monitorización de resultados durante finetuning.
* **Funciones de loss y métricas**: funciones de loss para masked language models y reply order prediction, métricas de evaluación para masked language models, así como las funciones de loss que aproximan métricas de evaluación para el finetuning de modelos TWilBert (cita LKE 2018).

Se pueden realizar tareas de entrenamiento, evaluación y finetuning mediante la llamada de los ficheros de aplicación, que reciben como parámetro ficheros de configuración json, donde se especifican los parámetros que definen modelos TWilBert. Estos ficheros de configuración se muestran con más detalle en las siguientes secciones. Sin embargo, no es obligatorio emplear dichas aplicaciones (fácilmente utilizables mediante la definición de ficheros de configuración) y se puede utilizar directamente el código proporcionado para implementar casos de uso no considerados por la herramienta e.g. modelos de fine-tuning más complejos.

# Estructura de TWilBert
***

El código de **TWilBert** se estructura en función de las utilidades que permite ejecutar la herramienta. En el directorio raiz se encuentran algunos scripts generales como las funciones de activación o las funciones de loss y métricas de evaluación para las etapas de entrenamiento y finetuning de modelos **TWilBert**. Junto con estos scripts, las herramientas de propósito más específico se agrupan en distintos paquetes:

* **applications**: scripts para ejecutar las tareas de entrenamiento, evaluación y finetuning de modelos **TWilBert**.
* **layers**: contiene varias capas utilizadas por los modelos, así como los encoders que las utilizan.
* **models**: modelo genérico de **TWilBert** para entrenar embeddings, así como algunos ejemplos de modelos para fine-tuning que emplean modelos pre-entrenados.
* **optimization**: técnicas de optimización para modelos **TWilBert** como learning rate annealers o optimizadores.
* **preprocessing**: funciones de preprocesado de texto y tokenizador.
* **scripts**: contiene scripts para realizar tareas requeridas para el entrenamiento/evaluación/finetuning de modelos **TWilBert**.
* **utils**: contiene funciones de utilidades para la preparación y generación de las muestras.
```
twilbert/
├── activations.py
├── applications
├── CONTRIBUTING.md
├── finetuning_losses.py
├── finetuning_monitor.py
├── pretraining_losses.py
├── pretraining_metrics.py
├── README.md
├── __init__.py
│   ├── __init__.py
│   ├── multiple_finetune_twilbert.py
│   ├── pretrain_twilbert.py
│   ├── single_finetune_twilbert.py
│   └── test_twilbert.py
├── layers
│   ├── encoders.py
│   ├── __init__.py
│   ├── layer_norm.py
│   ├── multihead_attention.py
│   ├── product_key_memory.py
│   └── self_attention.py
├── models
│   ├── finetuning_models.py
│   ├── __init__.py
│   └── twilbert_model.py
├── optimization
│   ├── __init__.py
│   ├── lr_annealing.py
│   └── optimizers.py
├── preprocessing
│   ├── __init__.py
│   └── tokenizer.py
├── scripts
│   └── create_vocab.py
└── utils
│   ├── generator.py
│   ├── __init__.py
│   └── utils.py
```
# Instalación
***
Descarga el repositorio de [**TWilBert**](LINK) en una carpeta dedicada para el proyecto.

```bash
mkdir project
cd project
git clone TWilBert
```
Los paquetes necesarios para ejecutarlo son:
* **Python 3.6.8**
* **Tensorflow 1.9.0**
* **Keras 2.2.4**
* **Pandas 0.22.2**
* **Numpy 1.17.3**
* **Scikit-Learn 0.21.1**
* [**Google SentencePiece**](https://github.com/google/sentencepiece)

Se recomienda crear las siguientes carpetas en la raiz de tu proyecto:
* **configs**: contendrá los ficheros de configuración para ejecutar tareas en **TWilBert**.
* **finetuning_tasks**: contendrá las downstream tasks con las que hacer finetuning.
* **finetuning_weights**: contendrá los pesos de los modelos una vez realizado finetuning.
* **pretraining_corpora**: contendrá los datasets para entrenar modelos **TWilBert**.
* **weights**: contendrá los pesos de los modelos TWilBert pre-entrenados.

Es necesario añadir la ruta de **TWilBert** al PATH de Python de la siguiente manera (o añadiéndola al fichero .bashrc):

```
export PYTHONPATH=$PYTHONPATH:/path/project/TWilBert
```

El primer paso para utilizar el framework consiste en obtener el vocabulario de la tarea con la que se ha pre-entrenado el modelo **TWilBert** a utilizar o con la que se quiere entrenar from scratch. Si usa los modelos pre-entrenados proporcionados por los autores, el fichero se puede descargar junto con los pesos de los modelos. Si no es el caso, es necesario generarlo utilizado la herramienta **/scripts/create_vocab.py** del framework.

```bash
python create_vocab.py ./pretraining_corpora/dataset.tsv
```
El fichero **dataset.tsv** es un fichero separado por tabuladores que contiene 4 columnas:

###### **`pretraining_corpora/dataset.tsv`**
```csv
ORIG_ID	ORIG_TEXT	REPLY_ID	REPLY_TEXT
1128609838655840256	De muy buena fuente tengo el conocimiento de que @jguaido propuso a @diegoarria como embajador ante la ONU y #AD, e… url	1128904660712955904	@munozoswaldo @TemplarioResisT @jguaido @DiegoArria Esto huele a FAKE NEWS..  GUAIDO NO SE DEJA POR LO CUAL ARRIA NO IBA  DE UNA
```

La primera y tercera columna son los identificadores de un tweet y su réplica respectivamente. La segunda y cuarta columna son los textos del tweet y de su réplica.


# Entrenamiento
***
***Advertencia***: no se recomienda entrenar modelos muy profundos desde cero si no se dispone de los recursos necesarios e.g. para un modelo **TWilBert** **L**=12, **A**=12, cross-sharing parameters, **E**=128, **H**=768, **PKM_KNN**=32, **PKM_MEMORY_SIZE**=256, **BS**=32 (gradient accumulation ***32**), **MAX_LEN**=64, **V**=30000, se requieren ~24GB de RAM.

Se asume que se dispone de un fichero .tsv como el mencionado en la sección anterior:

###### **`pretraining_corpora/dataset.tsv`**
```csv
ORIG_ID	ORIG_TEXT	REPLY_ID	REPLY_TEXT
1128609838655840256	De muy buena fuente tengo el conocimiento de que @jguaido propuso a @diegoarria como embajador ante la ONU y #AD, e… url	1128904660712955904	@munozoswaldo @TemplarioResisT @jguaido @DiegoArria Esto huele a FAKE NEWS..  GUAIDO NO SE DEJA POR LO CUAL ARRIA NO IBA  DE UNA
?   ?   ?   ?
...
?   ?   ?   ?
```

El primer paso necesario es crear el vocabulario como se ha especificado en la sección anterior. Con el vocabulario generado se debe definir el fichero de configuración json para el entrenamiento. Asumimos que se dispone del directorio **configs/train/** y se quiere definir un modelo denominado **xlarge**, por lo que se dispone del fichero **configs/train/config_xlarge.json**. Un ejemplo de definición del fichero es el siguiente:

###### **`configs/train/config_xlarge.json`**
```json
{
    "dataset": {

        "file": "./pretraining_corpora/short_urls_zz_pairs.tsv",
        "vocab_file": "vocab"
    },

    "representation": {

        "max_len": 64
    },

    "model": {

        "factorize_embeddings": true,
        "cross_sharing": true,
        "embedding_size": 128,
        "hidden_size": 768,
        "n_encoders": 12,
        "n_heads": 12,
        "attention_size": 64,
        "input_dropout": 0.0,
        "output_dropout": 0.0,
        "initializer_range": 0.02,
        "pkm": true,
        "pkm_params": {
            "factorize_embeddings": false,
            "k_dim": 512,
            "memory_size": 256,
            "n_heads": 4,
            "knn": 32,
            "in_layers": [10],
            "input_dropout": 0.0,
            "output_dropout": 0.0,
            "batch_norm": true
        },

        "masked_lm": {

            "type": "span",
            "max_span": 3,
            "budget": 0.15,
            "probs": {
                "mask": 0.8,
                "random": 0.1,
                "keep": 0.1
             }
         },

         "rop": {

             "n_hidden": 1,
             "hidden_size": 512
          }
    },

    "training": {

        "batch_size": 32,
        "epochs": 10,
        "optimizer": "adam",
        "noam_annealing": true,
        "warmup_steps": 8000,
        "accum_iters": 32,
        "use_gpu": true,
        "multi_gpu": true,
        "n_gpus": 2,
        "path_save_weights": "./weights/weights_xlarge/",
        "path_initial_weights" null,
        "verbose": 1
    }
}
```
En el campo **dataset** se definen las claves **file**, que contiene la ruta del dataset de entrenamiento con el formato especificado en la sección anterior y **vocab_file** que contiene la ruta del fichero con el vocabulario. El campo, **representation**, contiene parámetros para la representación de las muestras que tomará como entrada el modelo **TWilBert**. En este caso basta con especificar el parámetro **max_len** que define la longitud máxima de tweets y réplicas (notar que la longitud de entrada es (2 * **max_len**) + 3, debido a que se concatenan tweet y réplica y se añaden los tokens especiales **[MASK]**, **[SEP]** y **[CLS]**).

El campo **model** es el que contiene todos los hiper-parámetros del modelo a entrenar:

* **factorize_embeddings**: si se requiere desacoplar la dimensionalidad de los embeddings incontextuales de los contextuales.
* **cross_sharing**: si se requiere compartir los pesos de los transformer encoder a lo largo del modelo. Únicamente se ha implementado la estrategia [**ALL-SHARED**](https://arxiv.org/abs/1909.11942) donde todos los encoder comparten pesos.
* **embedding_size**: dimensionalidad de los embeddings incontextuales (**E**).
* **hidden_size**: dimensionalidad de los embeddings contextuales (**H**).
* **n_encoders**: número de transformer encoders del modelo (**L**).
* **n_heads**: número de cabezales de atención por encoder del modelo (**A**).
* **attention_size**: dimensionalidad **d_k** del mecanismo de multi-head attention. Si no se especifica se define como **hidden_size** / **n_heads**.
* **input_dropout**: probabilidad de dropout en la representación de entrada al modelo.
* **output_dropout**: probabilidad de dropout para las capas intermedias del modelo.
* **initializer_range**: desviación estándar de una distribución **truncated_normal** para la inicialización de los pesos.
* **pkm**: flag booleano para la utilización de Product Key Memory Layers (**PKM**).
* **pkm_params**: contiene la definición de los hiper-parámetros para las **PKM**:
  * **factorize_embeddings**: similar a [ALBERT] pero aplicada sobre la matriz de embeddings de las **PKM**.
  * **k_dim**: dimensionalidad de las queries y keys.
  * **memory_size**: tamaño de la memoria (nº de elementos en la matriz de embeddings).
  * **n_heads**: nº de cabezales, si >1 se suma la salida de todos los cabezales de la **PKM**.
  * **knn**: nº de vecinos más cercanos para la selección de keys.
  * **in_layers**: array con el número de capas del modelo en los que colocar 1 o más capas **PKM**.
  * **input_dropout**: probabilidad de dropout sobre la entrada de la **PKM**.
  * **output_dropout**: probabilidad de dropout sobre la salida de la **PKM**.
  * **batch_norm**: query batch normalization.
* **masked_lm**: contiene la definición de las propiedades del enmascarado de las muestras para aprender un masked language model (**MLM**). 
  * **type**: define el tipo de **MLM**. Si es **word** se enmascaran tokens individuales, si es **span** se enmascaran n-gramas de longitud máxima **max_span**. En cualquier caso se emplea **dynamic masking**.
  * **budget**: porcentaje sobre la entrada del número de tokens a enmascarar.
  * **probs**: probabilidades de cada tipo de enmascarado:
    * **mask**: probabilidad de enmascarar un token con el token especial **[MASK]**.
    * **random**: probabilidad de reemplazar un token por otro aleatorio del vocabulario.
    * **keep**: probabilidad de mantener el token.
* **rop**: contiene la definición del modelo, que se aplica sobre la salida del token **[CLS]**, para el problema de reply order prediction (**ROP**):
  * **n_hidden**: número de capas ocultas del modelo para **ROP**.
  * **hidden_size**: dimensionalidad de las capas ocultas.

El campo **training** contiene la definición de los parámetros de entrenamiento del modelo:
  * **batch_size**: número de muestras por batch.
  * **epochs**: número de épocas de entrenamiento.
  * **optimizer**: optimizador a utilizar. Son válidos todos los optimizadores de **Keras** y los proporcionados en el entorno: **adam** es una implementación de **ADAM** que permite gradient accumulation, **lamb** es una implementación de **LAMB** que permite gradient accumulation y también son válidos todos los optimizadores de **Keras** (si **optimizer**=**null**, el optimizador por defecto es la implementación de **ADAM** de **Keras**).
  * **noam_annealing**: flag para la utilización de **Noam** Learning Rate Annealing con **warmup_steps**.
  * **accum_iters**: número de acumulaciones para gradient accumulation. El batch size total es **batch_size** * **accum_iters**.
  * **use_gpu**: flag para especificar si se va a utilizar GPU para el entrenamiento.
  * **multi_gpu**: flag para especificar si se va a utilizar más de una GPU para el entrenamiento.
  * **n_gpus**: número de GPUs a utilizar en entrenamiento. Internamente se emplea un distribuidor de carga que emplea el número de GPUs para distribuir el cómputo de las capas del modelo.
  * **path_save_weights**: path donde guardar los pesos del modelo a cada época de entrenamiento.
  * **path_initial_weights**: path al fichero con los pesos para inicializar el modelo. Por defecto **null**, si se especifica un fichero, se inicializan los pesos del modelo con dichos pesos.
  * **verbose**: flag para indicar el tipo de verbosidad.

La llamada al script de entrenamiento con un fichero de configuración se muestra a continuación:

```bash
python3 twilbert/applications/pretrain_twilbert.py configs/train/config_xlarge.json
```

# Evaluación
***

Partiendo de un modelo ya entrenado y de un conjunto de muestras definidas en un fichero .tsv como en la sección previa, es posible observar su comportamiento a nivel de **MLM** y **ROP** sobre un conjunto de muestras de test. Un fichero de configuración de ejemplo para test se proporciona a continuación:

###### **`configs/test/config_xlarge.json`**
```json
{
    "dataset": {

        "test_file": "./pretraining_corpora/urls_test_zz_pairs.tsv",
        "vocab_file": "vocab"
    },

    "representation": {

        "max_len_training": 64,
        "max_len_test": 64
    },

    "model": {

        "factorize_embeddings": true,
        "cross_sharing": true,
        "embedding_size": 128,
        "hidden_size": 768,
        "n_encoders": 12,
        "n_heads": 12,
        "attention_size": 64,
        "input_dropout": 0.0,
        "output_dropout": 0.0,
        "initializer_range": 0.02,
        "pkm": true,
        "pkm_params": {
            "factorize_embeddings": false,
            "k_dim": 512,
            "memory_size": 256,
            "n_heads": 4,
            "knn": 32,
            "in_layers": [10],
            "input_dropout": 0.0,
            "output_dropout": 0.0,
            "batch_norm": true
        },

        "masked_lm": {

            "type": "span",
            "max_span": 3,
            "budget": 0.15,
            "probs": {
                "mask": 0.8,
                "random": 0.1,
                "keep": 0.1
             }
         },

         "rop": {

             "n_hidden": 1,
             "hidden_size": 512
          }
    },

    "test": {
        "path_load_weights": "./weights/weights_xlarge/model_02-5.1240-0.2600.hdf5"
    }
}
```

La definición de los campos **dataset** y **model** es idéntica al caso anterior (mirar la sección **"Entrenamiento"** para más información de los campos), donde ahora, **test_file** indica la ruta del fichero de test a utilizar. En la implementación del framework, se ha optado por mantener la definición de **model** utilizada en entrenamiento, también para los ficheros de configuración de test y finetuning, con el motivo de recordar y hacer explícito el tipo de modelo y los hiper-parámetros que se están empleando, reduciendo así el número de fallos. El campo **representation** ahora contiene los campos **max_len_training** y **max_len_test** siendo la longitud máxima de los textos empleados en entrenamiento y la longitud máxima a emplear en test respectivamente. Es conveniente notar que **max_len_test**<=**max_len_training** ya que el modelo no es capaz de generalizar para longitudes mayores. En el campo **test** se especifican propiedades del proceso de test, en este caso, únicamente **path_load_weights** que indica la ruta de los pesos de un modelo pre-entrenado.

La llamada al script de evaluación con un fichero de configuración se muestra a continuación:

```bash
python3 twilbert/applications/test_twilbert.py configs/test/config_xlarge.json
```

El script **test_twilbert.py** lleva a cabo un proceso de evaluación y visualización de los procesos que se aplican sobre la entrada, así como la salida del modelo pre-entrenado para **ROP** y **MLM**, con una muestra del fichero de test.

# Finetuning
***

Partiendo de modelos **TWilBert** pre-entrenados (como los proporcionados para Twitter en Español) se pueden emplear para abordar tareas de PLN. En el caso de finetuning se distinguen dos tipos de tareas, dependiendo de la entrada que el modelo reciba: **single-input tasks** y **multiple-input tasks**. Únicamente se diferencian por el tipo de entrada que los modelos reciben, o un único texto o varios.

El formato de los ficheros de finetuning depende del tipo de entrada. Si es una **single-input task**, el formato esperado por las aplicaciones del framework es un tsv con el identificador del texto, el texto y la clase de referencia separados por tabulador:

###### **`finetuning_tasks/single_task/set.tsv`**
```csv
ID      TEXT    CLASS
?   ?   ?
```

Si es una **multiple-input task**, el formato esperado es idéntico al anterior pero con más de una columna de texto (desde el fichero de configuración se seleccionan las dos columnas de texto a utilizar):

###### **`finetuning_tasks/multiple_task/set.tsv`**
```csv
ID  TEXT_1  TEXT_2  ...  TEXT_N   CLASS
?   ?   ?   ... ?   ?
```

Un fichero de configuración de ejemplo para **single-input tasks** se proporciona a continuación:

###### **`configs/finetuning/config_single_xlarge.json`**
```json
{
    "dataset": {

        "train_file": "./finetuning_tasks/stance17/train.csv",
        "dev_file": "./finetuning_tasks/stance17/dev.csv",
        "test_file": "./finetuning_tasks/stance17/dev.csv",
        "vocab_file": "vocab",
        "id_header": "ID",
        "text_header": "TEXT",
        "class_header": "CLASS",
        "delimiter": "\t"
    },

    "task":{

        "regression": false,
        "categories": {
           "AGAINST" : 0,
           "NEUTRAL": 1,
           "FAVOR": 2
        },
        "class_weights": {
           "0": 1,
           "1": 1,
           "2": 1
        },
        "eval_metric": "f1",
        "average_metric": "macro",
        "class_metric": null,
        "stance_f1": true,
        "multi_label": false
    },

    "representation": {

        "max_len": 64,
        "max_len_test": 64
    },

    "model": {

        "factorize_embeddings": true,
        "cross_sharing": true,
        "embedding_size": 128,
        "hidden_size": 768,
        "n_encoders": 12,
        "n_heads": 12,
        "attention_size": 64,
        "input_dropout": 0.0,
        "output_dropout": 0.0,
        "initializer_range": 0.02,
        "pkm": true,
        "pkm_params": {
            "factorize_embeddings": false,
            "k_dim": 512,
            "memory_size": 256,
            "n_heads": 4,
            "knn": 32,
            "in_layers": [10],
            "input_dropout": 0.0,
            "output_dropout": 0.0,
            "batch_norm": true
        },

         "rop": {

             "n_hidden": 1,
             "hidden_size": 512
          }
    },

    "finetuning": {

        "batch_size": 32,
        "pred_batch_size": 32,
        "epochs": 10,
        "finetune_all": true,
        "collapse_mode": "avg",
        "use_special_tokens": true,
        "loss": "categorical_crossentropy",
        "optimizer": "adam_accumulated",
        "lr": 0.0001,
        "noam_annealing": false,
        "warmup_steps": 5,
        "accum_iters": 1,
        "use_gpu": true,
        "multi_gpu": true,
        "n_gpus": 2,
        "path_save_weights": "./finetuning_weights/stance17/",
        "path_load_weights": "./weights/weights_xlarge/model_02-5.1240-0.2600.hdf5",
        "model_name": "stance_model",
        "verbose": 1
    }
}

```

***Recordatorio***: a pesar de que el entorno permite la utilización de modelos **TWilBert** mediante ficheros de configuración, el usuario final es libre de emplear el entorno directamente desde sus propios scripts Python para ampliar la funcionalidad a casos de uso no contemplados.

De nuevo, el campo **model** es idéntico a la definición en el caso de entrenamiento y evaluación de los modelos. También, el campo **representation** se define igual que en el caso de la evaluación de los modelos (recordar que **max_len_test**<=**max_len**). El campo **dataset** es diferente, donde ahora se especifican los siguientes campos:

* **train_file**: fichero de entrenamiento para finetuning.
* **dev_file**: fichero de validación para finetuning.
* **test_file**: fichero de test para finetuning.
* **id_header**: nombre de la columna para el identificador (**ID** en el fichero de ejemplo)
* **text_header**: nombre de la columna para el texto (**TEXT** en el fichero de ejemeplo)
* **class_header**: nombre de la columna para las referencias (**CLASS** en el fichero de ejemplo)
* **delimiter**: tipo de delimitador del fichero (por defecto **\t**)

El campo **task** se utiliza para especificar los detalles de la tarea con la que se va a realizar el finetuning:

* **regression**: flag para especificar si es una tarea de regresión.
* **categories**: mapping de las clases de referencia a un valor entero desde 0 a |**C**|-1.
* **class_weights**: pesos para ponderar la loss de cada clase (cost-sensitive training para corpus desbalanceados)
* **eval_metric**: métrica de evaluación de la tarea (**f1**, **accuracy**, **precision**, **recall**, **pearson**)
* **average_metric**: tipo de ponderación de la métrica de evaluación (**macro**, **micro**, **null**)
* **class_metric**: evaluación independiente para una clase concreta (e.g. f1 de la clase 1 -> **class_metric**=1)
* **stance_f1**: variante de la F1 donde se obvia una de las clases.
* **multi_label**: flag para especificar si la tarea es multi-label

En el campo **finetuning** se especifican algunos hiper-parámetros del modelo (que se aplica sobre la salida de los modelos **TWilBert**, por defecto una única capa softmax) y del proceso de finetuning. El usuario puede definir cualquier modelo que requiera, siguiendo el modelo de ejemplo proporcionado en el fichero **/twilbert/models/finetuning_models.py**):

* **batch_size**: tamaño de batch para finetuning.
* **pred_batch_size**: tamaño de batch para la evaluación del modelo en cada época de finetuning.
* **epochs**: número de épocas de finetuning.
* **finetune_all**: flag para especificar si el modelo **TWilBert** se entrena durante finetuning o no.
* **collapse_mode**: especifica la manera de obtener una representación vectorial a partir de la entrada a **TWilBert** (**avg**, **cls**)
* **use_special_tokens**: flag para considerar los tokens especiales **[CLS]** y **[SEP]** cuando **collapse_mode**=**avg**.
* **loss**: función de loss para finetuning, se puede especificar cualquier función de loss integrada en Keras o las funciones de loss propuestas en (**macro_f1**, **macro_precision**, **macro_recall**, **f1_macro**, **micro_f1**, **micro_precision**, **micro_recall**, **jaccard_acc**). Si usas alguna de estas funciones de loss, referencia esta [publicación](https://content.iospress.com/articles/journal-of-intelligent-and-fuzzy-systems/ifs179019).
* **optimizer**: optimizador (**adam_accumulated**, **lamb_accumulated** o cualquier optimizador de Keras)
* **lr**: learning rate.
* **noam_annealing**: flag para especificar si se emplea Noam durante finetuning, con **warmup_steps**.
* **accum_iters**: número de iteraciones para gradient accumulation.
* **use_gpu**: flag para especificar si se emplea GPU.
* **multi_gpu**: flag para especificar si se emplean varias GPUs.
* **n_gpus**: número de GPUs a utilizar.
* **path_save_weights**: ruta donde guardar los pesos de los modelo a cada época de finetuning.
* **path_load_weights**: ruta desde donde cargar los pesos de un modelo TWilBert pre-entrenado.

***Recomendación***: se recomienda regularizar correctamente los modelos de finetuning para evitar overfitting e.g. mediante dropout y utilizar el mismo optimizador que durante el entrenamiento (**adam_accumulated** con los modelos pre-entrenados) con learning rates entre 1e-3 y 1e-5.

El fichero de configuración para **multiple-input tasks** es idéntico al de **single-input tasks** pero especificando el parámetro **aux_header** dentro del campo **dataset**, dándole el valor de la columna donde se ubica, en los ficheros de entrada, el campo de texto auxiliar (segunda entrada al modelo **TWilBert**).

La llamada del script para hacer finetuning de modelos **TWilBert** se muestra a continuación:

```bash
python3 twilbert/applications/single_finetune_twilbert.py configs/finetuning/config_single_xlarge.json
```

Tras ejecutar el script, se lleva a cabo el finetuning de un modelo **TWilBert** pre-entrenado, junto con un proceso de monitorización automática de los resultados siguiendo los parámetros de evaluación definidos en el campo **task**.


# Modelos Pre-entrenados
***
Junto con la implementación del framework, se proporcionan varios modelos pre-entrenados.

***Advertencia***: actualmente solo está disponible el modelo **XLarge**. En el futuro publicaremos el resto de modelos, aunque valores de las tablas están sujetos a cambios. Todos los modelos han sido pre-entrenados con **140** millones de pares de tweets (**70m** positivos y **70m** negativos) durante 3 épocas.

|         | Link |  L |  A | E   | H   | Cross | KNN | M   | Dk  | J | Params |
|---------|-----:|---:|---:|-----|-----|-------|-----|-----|-----|---|--------|
| Base    |    ? |  6 |  6 | 128 | 768 | Yes   | 32  | 256 | 512 | 4 | 89m    |
| Large   |    ? | 10 | 10 | 128 | 768 | Yes   | 32  | 256 | 512 | 4 | 89m    |
| XLarge  |    ? | 12 | 12 | 128 | 768 | Yes   | 32  | 256 | 512 | 4 | 89m    |
| XXLarge |    ? | 16 | 16 | 128 | 768 | Yes   | 32  | 256 | 512 | 4 | 89m    |


# Resultados
***

Se ha evaluado el modelo **XLarge** sobre gran cantidad de tareas en Twitter para español de congresos internacionales como la **SEPLN** y **SemEval**. Las tareas de clasificación de textos con las que se ha evaluado son: Topic Detection ([COSET 2017](http://mediaflows.es/coset/)), Irony Detection ([IroSVA 2019](https://www.autoritas.net/IroSvA2019/)), Gender Detection ([Stance&Gender 2017](http://stel.ub.edu/Stance-IberEval2017/)), Humor Detection ([HAHA 2019](https://www.fing.edu.uy/inco/grupos/pln/haha/)), Sentiment Analysis ([TASS 2019](https://sites.google.com/view/iberlef-2019)), Emotional Categorization ([TASS 2018 Task 4](http://www.sepln.org/workshops/tass/2018/task-4/)), Hate Speech Detection ([HATEval 2019]((https://competitions.codalab.org/competitions/19935)) & [Mex-A3T 2019](https://sites.google.com/view/mex-a3t/)), Stance Detection ([Stance 2017](http://stel.ub.edu/Stance-IberEval2017/) & [MultiStance 2018](https://www.autoritas.net/MultiStanceCat-IberEval2018/)) y Affect Detection ([SemEval 2018 Task 1](https://competitions.codalab.org/competitions/17751))


|               | MP | MR | MF1 | F1 | Acc | MP | MR | MF1 | F1 | Acc |
|---------------|---:|---:|----:|----|-----|----|----|-----|----|-----|
| Coset17       |  |    |     |    |     | -   |  -  |  64.82  |  -  | -    |
| IroSVA19-ES      |    |    |     |    |     |  -  | -   | 71.67    | -   | -     |
| IroSVA19-MX      |    |    |     |    |     |  -  |  -  | 68.03    |  -  |  -   |
| IroSVA19-CU      |    |    |     |    |     |  -  | -   | 65.96    |   -  |  -   |
| Gender17      |    |    |     |    |  73.08   |-    |  -  |     -|  -  |  68.55    |
| HAHA19        |    |    |     |    |     |  -  | -   |    - |  82.10  |   85.50  |
| TASS19-ES        |    |    |     |    |     |  50.50  | 50.80   |  50.70   |  -  | -    |
| TASS19-MX        |    |    |     |    |     |  49.00  |  51.20  |  50.10   |  -  |  -   |
| TASS19-PE        |    |    |     |    |     |   46.20 | 44.60   |  45.40   |  -  |  -   |
| TASS19-CR        |    |    |     |    |     |  58.80  | 45.40   | 51.20    | -   |  -   |
| TASS19-UY        |    |    |     |    |     |  49.70   | 53.60   |  51.50   |  -  | -    |
| TASS18-4      |    |    |     |    |     |  87.80 |  88.90  |   88.30  |  -  |  89.30   |
| HATEval19     |    |    |     |    |     |    |    |     |    |     |
| MEX-A3T19     |    |    |     |    |     |    |    |     |    |     |
| Stance17      |    |    |     |    |     | -   | -   |  48.88   |  -  |  -   |
| MultiStance18 |    |    |     |    |     |  -  |  -  |  28.02   |  -  |  -   |
| SemEval18-1   |    |    |     |    |     |    |    |     |    |     |
