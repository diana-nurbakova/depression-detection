# Qualitative Task 1 Case Studies

Run: `Run 2 (A1 lightweight)` (run_idx=2)
Selection: top 2 per instrument by |replay_MAE − submitted_MAE|

## PHQ-9

### S06 — PHQ-9

- Gold band: `moderately_severe` (total 16)
- Submitted (round 30): total 11, MAE_items=1.000
- Replay (round 16): total 13, MAE_items=0.778
- |ΔMAE| = 0.222

**Items (pred at last available round vs gold):**

| Item | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pred (submitted) | 2 | 2 | 1 | 1 | 1 | 2 | 1 | 1 | 0 | 11 |
| gold | 2 | 3 | 1 | 3 | 1 | 1 | 2 | 0 | 3 | 16 |

| Item | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pred (replay) | 2 | 2 | 1 | 2 | 1 | 2 | 2 | 1 | 0 | 13 |
| gold | 2 | 3 | 1 | 3 | 1 | 1 | 2 | 0 | 3 | 16 |

**LLM reasoning at submitted round 30 (CoT excerpt):**

_(no CoT logged)_

**LLM reasoning at replay round 16 (CoT excerpt):**

- **1. Anhedonia** (score=2): lo último que me apetece, porque básicamente no quiero estar allí
- **2. Depressed mood** (score=2): me siento amargado, inutilidad existencial
- **3. Sleep disturbance** (score=2): hay días que estoy deseando llegar a casa para poder dormir aunque sea media hora
- **4. Fatigue** (score=2): estoy cansado
- **5. Appetite** (score=1): no mention of appetite changes
- **6. Worthlessness** (score=2): inutilidad existencial, me esfuerzo mucho en la vida, y en realidad no tiene ningún sentido
- **7. Concentration** (score=2): no puedo concentrarme en nada
- **8. Psychomotor** (score=2): me quita un poco las fuerzas para seguir adelante
- **9. Suicidal ideation** (score=0): no mention of suicidal thoughts or desires

**Transcript slice (patient turns; therapist turns where logged):**

- **R12** patient: _Yo creo que es la falta de tiempo_
  - therapist (prev): _Parece que hay algunas cosas que te gustan. ¿Qué crees que te está bloqueando para hacer esas cosas ahora? ¿Es solo la falta de tiempo o hay algo más?_
- **R13** patient: _Ahí soy feliz. Básicamente pienso "ahora voy a descansar". Y me olvido del mundo. Son los momentos en los que estoy bien, que son muy pocos El problema es el resto del tiempo, que es cuando estoy amargado_
  - therapist (prev): _¿Y qué te dice tu mente cuando piensas en tomarte un rato para jugar, ver una película o leer? ¿Qué pensamientos o sensaciones surgen?_
- **R14** patient: _Depende mucho de la situación. En el trabajo intento disociar y que se pase el tiempo lo antes posible, básicamente porque no quiero estar allí. Así que intento concentrarme en la tarea y no prestar atención al reloj para no agobiarme El problema es que eso pues no es vivir, pero tampoco sabría como_
  - therapist (prev): _¿Qué pasa con el resto del tiempo, cuando te sientes amargado? ¿Qué pensamientos o sensaciones te acompañan?_
- **R15** patient: _Pues básicamente es como que yo siento que me esfuerzo mucho en la vida, y en realidad no tiene ningún sentido, porque al final me voy a morir igualmente O sea, podría no hacer absolutamente nada y el resultado final sería el mismo O podría estar esforzándome mucho para que mañana me atropellase un _
  - therapist (prev): _Parece que la sensación de inutilidad existencial está presente en muchas áreas de tu vida, ¿verdad? ¿Qué te dice esa voz interior cuando te sientes así?_
- **R16** patient: _Pues me quita un poco las fuerzas para seguir adelante_
  - therapist (prev): _Entiendo que te sientes como si estuvieras luchando contra una pared, y que no importa lo que hagas, el resultado final es lo mismo.  ¿Cómo te afecta ese pensamiento de que nada tiene sentido si te vas a morir?_
- **R25** patient: _Buenas. Pues un poco igual que la otra vez, no sé. No ha cambiado nada._
  - therapist (prev): _Perfecto, Manuel. Nos vemos el viernes. Cuidate._
- **R26** patient: _No sé. Sigo igual. Misma rutina, mismas cosas..._
  - therapist (prev): _¿Podrías contarme un poco más sobre qué es lo que sientes que no ha cambiado?_
- **R27** patient: _Pues no lo sé, la verdad. Es que tampoco tengo muy claro qué opciones de cambio real tengo. Siento que la mayoría de mi tiempo gira en torno a estar bien para poder cumplir con las obligaciones_
  - therapist (prev): _Entiendo ¿Te has parado a pensar qué es lo que te gustaría cambiar de esa rutina o hacer diferente en tu vida? ¿Algo que te haría sentir más vivo o más conectado con lo que te importa?_
- **R28** patient: _No entiendo bien la pregunta. No ocurre nada en particular. O no sé si podría decir que estoy "atascado" en esa idea_
  - therapist (prev): _¿Y qué ocurre cuando te quedas atascado en esa idea de "estar bien para cumplir"?_
- **R29** patient: _Eso sí. Así es como me siento. Y claro, me entristece. Porque qué sentido tiene vivir si al final se resume en ser funcional para el sistema pero no lo disfruto de verdad? Pero tampoco es como que pueda escapar de eso._
  - therapist (prev): _Me refiero a si te sientes atrapado en la idea de tener que estar bien para cumplir con tus obligaciones, como si te estuvieras pasando la vida tratando de mantener ese estado y no pudieras escapar de él. ¿Te suena un poco a eso?_
- **R30** patient: _Lo veo como algo generalizado_
  - therapist (prev): _Es normal que te preocupe sentir que tu vida se reduce a cumplir obligaciones y no a disfrutarla. ¿Qué te parece si exploramos un poco esa sensación de "no poder escapar"? ¿Crees que te sientes atrapado por algo específico, o es más una sensación generalizada?_
- **R31** patient: _No creo que haga nada en particular para lidiar con esa sensación, porque no creo que se pueda hacer nada. Sigo con mi vida, pero un poco más triste_
  - therapist (prev): _¿Te has dado cuenta de cuando te enfrentas a esa sensación, ¿qué haces normalmente para lidiar con ella?_
- **R32** patient: _Ahí es cuando me pongo más triste y me vienen pensamientos de querer morirme._
  - therapist (prev): _Entiendo. Es como si te resignaras a llevar esa tristeza contigo. ¿Y qué te dice tu mente cuando te enfrentas a esa sensación de tristeza y "no poder hacer nada"? ¿Qué tipo de pensamientos te surgen?_
- **R33** patient: _Ahora mismo no. En el pasado sí que lo intenté en alguna ocasión, pero hace más de 10 años de aquello. En ese sentido estoy "tranquilo", pero a la vez me preocupa un poco acabar llegando al mismo punto._
  - therapist (prev): _Lo siento mucho, Manuel.  ¿Podrías decirme más sobre esos pensamientos de querer morirte? ¿Te planteas hacerles caso?_
- **R34** patient: _Tampoco lo tengo muy claro. Ahora mismo me da miedo la muerte, y eso hace mucho._
  - therapist (prev): _Entiendo, muchas gracias por compartirlo conmigo. ¿Qué es lo que te impulsa a mantenerte a pesar de esos pensamientos?_
- **R35** patient: _Dejar de existir me abruma A veces intento imaginarlo y me da la ansiedad. En plan, que se me acelera el corazón y lloro. Porque es imposible imaginar como es no existir. Es como imaginar qué pasaba antes de haber nacido, y no sé. Me da como crisis existencial_
  - therapist (prev): _Es normal que tengas miedo a la muerte. ¿Sabes qué es lo que te da miedo de la muerte? ¿Es el no saber qué hay después, el dejar a tus seres queridos, o algo más?_

**Commentary:** _(fill in)_

---

### S04 — PHQ-9

- Gold band: `moderate` (total 13)
- Submitted (round 30): total 10, MAE_items=0.778
- Replay (round 17): total 12, MAE_items=0.556
- |ΔMAE| = 0.222

**Items (pred at last available round vs gold):**

| Item | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pred (submitted) | 1 | 2 | 1 | 1 | 1 | 2 | 1 | 1 | 0 | 10 |
| gold | 2 | 1 | 1 | 2 | 1 | 1 | 3 | 2 | 0 | 13 |

| Item | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pred (replay) | 1 | 2 | 1 | 1 | 1 | 2 | 2 | 2 | 0 | 12 |
| gold | 2 | 1 | 1 | 2 | 1 | 1 | 3 | 2 | 0 | 13 |

**LLM reasoning at submitted round 30 (CoT excerpt):**

_(no CoT logged)_

**LLM reasoning at replay round 17 (CoT excerpt):**

_(no CoT logged)_

**Transcript slice (patient turns; therapist turns where logged):**

- **R13** patient: _Se niega a recibirlo, solo tiene una medicación que le mandaron hace muchísimos años, se limita a tomarse las pastillas y su actividad es nula._
  - therapist (prev): _Por lo que me cuentas, entiendo que por la situación de tu madre quizás has estado muy pendiente de ella. ¿Está recibiendo tratamiento psicológico o psiquiátrico?_
- **R14** patient: _Desde niña intentar constantemente que eso cambiará y que dentro de la complejidad del problema la ayuda pudiese hacer que llevase una vida mejor o que pudiésemos llevarla juntas. Pero cuando todos los intentos fracasaron, me limite a acompañarla en su estado, a veces pasando a ser una sombra de ese_
  - therapist (prev): _¿Cuando vivias con ellos , qué hacías tú en relación a ello? ¿Como es la relación con tu madre en concreto? Disculpa que te haga tantas preguntas, es que necesito entender para ver cómo te puedo ayudar_
- **R15** patient: _Sí, es exactamente la palabra que usaría, resentimiento. Es un sentimiento que ha subido progresivamente durante los años, de modo que a veces encuentro que esos momentos cuando salta todo por los aires son provocados por mí, como si ya no los aguantase, o tuviese algo dentro pudriéndose que me hace_
  - therapist (prev): _Comprendo. Parece una dinámica familiar compleja y agotadora, y es natural que te sientas atrapada entre el cariño y la frustración. Parece que has asumido un rol de cuidadora desde muy pequeña, incluso a costa de tu propio bienestar. ¿Te has sentido resentida en algún momento por ello?_
- **R16** patient: _Sí completamente culpable, porque ese resentimiento suele crear conflictos, y yo no llevo nada bien estar mal con ellos._
  - therapist (prev): _Es muy valiente reconocer eso. El resentimiento es una emoción muy compleja y, a menudo, difícil de expresar. Parece que has estado conteniendo ese sentimiento durante mucho tiempo, y que ahora está saliendo a la superficie de formas que quizás no te gustan. ¿Te has sentido culpable por sentirte así_
- **R17** patient: _No sé si es por la culpa o porque aunque hay una comunicación total, hay una inexistente comunicación emocional, pero no soy capaz de expresarlo con ellos. En cuanto a los pensamientos negativos, no tengo duda de que sí me hace más vulnerable a ellos._
  - therapist (prev): _Es importante reconocer esa culpa, pero también es esencial entender que sentir resentimiento no significa que seas mala persona. Es una emoción legítima y, a veces, necesaria para sanar. ¿Crees que esa culpa te impide expresar lo que sientes o te hace más vulnerable a esos pensamientos negativos?_
- **R25** patient: _No tengo muchas aficiones, cuando estoy sola ver series o leer.  Pero lo que más disfruto, es pasar tiempo con mis amigas o mi pareja, haciendo cualquier cosa._
  - therapist (prev): _Gracias por la información. A veces, nos pasa eso, que no podemos concentrarnos en las tareas y más si conllevan un esfuerzo de atención y concentración. ¿Sueles dedicar tiempo a algo que te guste, hobby?_
- **R26** patient: _Sí eso me despeja, tengo total confianza con ellos, les suelo contar mi día a día y lo que me preocupa, el tema que hablamos ayer del miedo a la muerte o que pasé algo no lo hablo tan a menudo, porque tampoco quiero molestarles demasiado con algo que no tiene solución._
  - therapist (prev): _Me alegro que me digas que te gusta estar con gente cercana a ti como tus amigas y pareja ¿Qué confianza tienes con ellos? ¿les cuentas cómo te sientes? ¿lo que te preocupa?_
- **R27** patient: _No mucho, se limitan a escuchar y a validar lo que siento. Trato de evitar hablarlo últimamente porque me produce miedo que me de un ataque como otras veces si lo pienso de más._
  - therapist (prev): _Entiendo. A veces, puede ser difícil compartir cosas que nos parecen pesadas o que parecen que no tienen solución, pero es cierto que a menudo nos sentimos aliviados al hablar de ellas con alguien que nos importa. ¿Cuando a veces, les has comentado algo al respecto, ¿Cómo han reaccionado? ¿qué te su_
- **R28** patient: _cuando lo pienso demasiado llega un momento en que es como si el miedo se desbordase, es una sensación de terror intenso como si sintiese que me estoy muriendo, dura unos segundos y luego me quedo muy cansada._
  - therapist (prev): _¿cuando lo hablas te dan ataques? dime más sobre esto por favor_
- **R29** patient: _Noto falta de aire y una presión fuerte en el pecho, mareo y un miedo muy muy intenso, más del que allá sentido en cualquier otra ocasión de mi vida.  Los pensamientos son relacionados con el miedo a morir, la primera vez que me pasó tenía 11 años y como no lo conocía tuve la sensación de que estaba_
  - therapist (prev): _Vaya, entiendo que si eso te ha pasado, intentes evitar que suceda de nuevo. Si no te importa, y sabiendo que es algo que no te gusta y como te he dicho, intentas evitar, por favor, descríbeme lo que te pasa. Indícame qué sensaciones físicas tienes (qué notas en cuerpo), qué pensamientos y/o imágene_
- **R30** patient: _Esa sensación solo segundos, es horrible pero muy breve_
  - therapist (prev): _¿Cuanto tiempo te dura?_
- **R31** patient: _Estaba en la cama antes de dormir, fue porque empecé a tener consciencia de la muerte y lo que significaba, estaba pensando en ello._
  - therapist (prev): _Me dices que el primer ataque fue a los 11 años. ¿Recuerdas donde estabas y qué pasó que pudiera ser el desencadentante?_
- **R32** patient: _Sí, siempre es cuando pienso en la muerte_
  - therapist (prev): _sí es cierto, aunque sean segundos, la sensación es muy muy desagradable ¿Te sigue pasando ahora igual, es cuando piensas en la muerte?_
- **R33** patient: _Pienso en la nada que hay después, no soy creyente así que es complicado aceptar y relacionarme con el concepto de la muerte. Pienso en el vacío, en la nada y en mis seres queridos._
  - therapist (prev): _Cuéntame un poco más de los pensamientos concretos que tienes_
- **R34** patient: _En el hecho de que cuando ocurra su muerte o la mía, no les veré más._
  - therapist (prev): _Sobre tus seres queridos qué piensas?_
- **R35** patient: _Para mí se traduce en soledad, soy una persona que valora mucho las relaciones sociales con la gente que verdaderamente me importa. Son un punto clave de mi vida, por eso pienso en cuando falten o en cuando yo muera, aunque lo segundo no debería preocuparme porque ya no existiré, no puedo evitar que_
  - therapist (prev): _Y si no los ves más, eso qué significa para ti?_

**Commentary:** _(fill in)_

---

## GAD-7

### S06 — GAD-7

- Gold band: `moderate` (total 10)
- Submitted (round 30): total 5, MAE_items=1.000
- Replay (round 16): total 9, MAE_items=0.429
- |ΔMAE| = 0.571

**Items (pred at last available round vs gold):**

| Item | 1 | 2 | 3 | 4 | 5 | 6 | 7 | total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pred (submitted) | 0 | 1 | 2 | 0 | 0 | 1 | 1 | 5 |
| gold | 2 | 2 | 2 | 2 | 1 | 1 | 0 | 10 |

| Item | 1 | 2 | 3 | 4 | 5 | 6 | 7 | total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pred (replay) | 1 | 2 | 2 | 2 | 0 | 1 | 1 | 9 |
| gold | 2 | 2 | 2 | 2 | 1 | 1 | 0 | 10 |

**LLM reasoning at submitted round 30 (CoT excerpt):**

_(no CoT logged)_

**LLM reasoning at replay round 16 (CoT excerpt):**

- **1. Nervousness** (score=None): _no evidence_
- **2. Uncontrollable worry** (score=None): _no evidence_
- **3. Excessive worry** (score=None): _no evidence_
- **4. Trouble relaxing** (score=None): _no evidence_
- **5. Restlessness** (score=None): _no evidence_
- **6. Irritability** (score=None): _no evidence_
- **7. Fear** (score=None): _no evidence_

**Transcript slice (patient turns; therapist turns where logged):**

- **R12** patient: _Yo creo que es la falta de tiempo_
  - therapist (prev): _Parece que hay algunas cosas que te gustan. ¿Qué crees que te está bloqueando para hacer esas cosas ahora? ¿Es solo la falta de tiempo o hay algo más?_
- **R13** patient: _Ahí soy feliz. Básicamente pienso "ahora voy a descansar". Y me olvido del mundo. Son los momentos en los que estoy bien, que son muy pocos El problema es el resto del tiempo, que es cuando estoy amargado_
  - therapist (prev): _¿Y qué te dice tu mente cuando piensas en tomarte un rato para jugar, ver una película o leer? ¿Qué pensamientos o sensaciones surgen?_
- **R14** patient: _Depende mucho de la situación. En el trabajo intento disociar y que se pase el tiempo lo antes posible, básicamente porque no quiero estar allí. Así que intento concentrarme en la tarea y no prestar atención al reloj para no agobiarme El problema es que eso pues no es vivir, pero tampoco sabría como_
  - therapist (prev): _¿Qué pasa con el resto del tiempo, cuando te sientes amargado? ¿Qué pensamientos o sensaciones te acompañan?_
- **R15** patient: _Pues básicamente es como que yo siento que me esfuerzo mucho en la vida, y en realidad no tiene ningún sentido, porque al final me voy a morir igualmente O sea, podría no hacer absolutamente nada y el resultado final sería el mismo O podría estar esforzándome mucho para que mañana me atropellase un _
  - therapist (prev): _Parece que la sensación de inutilidad existencial está presente en muchas áreas de tu vida, ¿verdad? ¿Qué te dice esa voz interior cuando te sientes así?_
- **R16** patient: _Pues me quita un poco las fuerzas para seguir adelante_
  - therapist (prev): _Entiendo que te sientes como si estuvieras luchando contra una pared, y que no importa lo que hagas, el resultado final es lo mismo.  ¿Cómo te afecta ese pensamiento de que nada tiene sentido si te vas a morir?_
- **R25** patient: _Buenas. Pues un poco igual que la otra vez, no sé. No ha cambiado nada._
  - therapist (prev): _Perfecto, Manuel. Nos vemos el viernes. Cuidate._
- **R26** patient: _No sé. Sigo igual. Misma rutina, mismas cosas..._
  - therapist (prev): _¿Podrías contarme un poco más sobre qué es lo que sientes que no ha cambiado?_
- **R27** patient: _Pues no lo sé, la verdad. Es que tampoco tengo muy claro qué opciones de cambio real tengo. Siento que la mayoría de mi tiempo gira en torno a estar bien para poder cumplir con las obligaciones_
  - therapist (prev): _Entiendo ¿Te has parado a pensar qué es lo que te gustaría cambiar de esa rutina o hacer diferente en tu vida? ¿Algo que te haría sentir más vivo o más conectado con lo que te importa?_
- **R28** patient: _No entiendo bien la pregunta. No ocurre nada en particular. O no sé si podría decir que estoy "atascado" en esa idea_
  - therapist (prev): _¿Y qué ocurre cuando te quedas atascado en esa idea de "estar bien para cumplir"?_
- **R29** patient: _Eso sí. Así es como me siento. Y claro, me entristece. Porque qué sentido tiene vivir si al final se resume en ser funcional para el sistema pero no lo disfruto de verdad? Pero tampoco es como que pueda escapar de eso._
  - therapist (prev): _Me refiero a si te sientes atrapado en la idea de tener que estar bien para cumplir con tus obligaciones, como si te estuvieras pasando la vida tratando de mantener ese estado y no pudieras escapar de él. ¿Te suena un poco a eso?_
- **R30** patient: _Lo veo como algo generalizado_
  - therapist (prev): _Es normal que te preocupe sentir que tu vida se reduce a cumplir obligaciones y no a disfrutarla. ¿Qué te parece si exploramos un poco esa sensación de "no poder escapar"? ¿Crees que te sientes atrapado por algo específico, o es más una sensación generalizada?_
- **R31** patient: _No creo que haga nada en particular para lidiar con esa sensación, porque no creo que se pueda hacer nada. Sigo con mi vida, pero un poco más triste_
  - therapist (prev): _¿Te has dado cuenta de cuando te enfrentas a esa sensación, ¿qué haces normalmente para lidiar con ella?_
- **R32** patient: _Ahí es cuando me pongo más triste y me vienen pensamientos de querer morirme._
  - therapist (prev): _Entiendo. Es como si te resignaras a llevar esa tristeza contigo. ¿Y qué te dice tu mente cuando te enfrentas a esa sensación de tristeza y "no poder hacer nada"? ¿Qué tipo de pensamientos te surgen?_
- **R33** patient: _Ahora mismo no. En el pasado sí que lo intenté en alguna ocasión, pero hace más de 10 años de aquello. En ese sentido estoy "tranquilo", pero a la vez me preocupa un poco acabar llegando al mismo punto._
  - therapist (prev): _Lo siento mucho, Manuel.  ¿Podrías decirme más sobre esos pensamientos de querer morirte? ¿Te planteas hacerles caso?_
- **R34** patient: _Tampoco lo tengo muy claro. Ahora mismo me da miedo la muerte, y eso hace mucho._
  - therapist (prev): _Entiendo, muchas gracias por compartirlo conmigo. ¿Qué es lo que te impulsa a mantenerte a pesar de esos pensamientos?_
- **R35** patient: _Dejar de existir me abruma A veces intento imaginarlo y me da la ansiedad. En plan, que se me acelera el corazón y lloro. Porque es imposible imaginar como es no existir. Es como imaginar qué pasaba antes de haber nacido, y no sé. Me da como crisis existencial_
  - therapist (prev): _Es normal que tengas miedo a la muerte. ¿Sabes qué es lo que te da miedo de la muerte? ¿Es el no saber qué hay después, el dejar a tus seres queridos, o algo más?_

**Commentary:** _(fill in)_

---

### S07 — GAD-7

- Gold band: `mild` (total 7)
- Submitted (round 30): total 7, MAE_items=0.286
- Replay (round 16): total 11, MAE_items=0.857
- |ΔMAE| = 0.571

**Items (pred at last available round vs gold):**

| Item | 1 | 2 | 3 | 4 | 5 | 6 | 7 | total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pred (submitted) | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 7 |
| gold | 1 | 1 | 1 | 2 | 0 | 1 | 1 | 7 |

| Item | 1 | 2 | 3 | 4 | 5 | 6 | 7 | total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pred (replay) | 2 | 2 | 2 | 1 | 1 | 1 | 2 | 11 |
| gold | 1 | 1 | 1 | 2 | 0 | 1 | 1 | 7 |

**LLM reasoning at submitted round 30 (CoT excerpt):**

_(no CoT logged)_

**LLM reasoning at replay round 16 (CoT excerpt):**

- **1. Nervousness** (score=3): The patient describes feeling 'muy nerviosa' before exposures and physical symptoms like sweating and palpitations, indicating nervousness a
- **2. Uncontrollable worry** (score=3): The patient's description of always feeling nervous before exposures and the persistence of anxiety during them suggests difficulty in contr
- **3. Excessive worry** (score=1): While the patient worries about exposures, there's no clear indication of worrying about multiple different things beyond this context.
- **4. Trouble relaxing** (score=2): The patient mentions not being able to fully relax until after the exposure and even then, not completely, suggesting difficulty in relaxing
- **5. Restlessness** (score=2): The physical symptoms and the description of the anxiety experience during exposures suggest restlessness and an inability to sit still calm
- **6. Irritability** (score=1): There's no direct mention of the patient becoming easily annoyed or angry with others.
- **7. Fear** (score=3): The patient's fear of being judged and making mistakes, and the overall anxiety experience, suggest a fear that something terrible could hap

**Transcript slice (patient turns; therapist turns where logged):**

- **R12** patient: _Vale, perfecto. Adiós!_
  - therapist (prev): _De acuerdo, pues nos vemos si quieres la semana que viene. Intenta estos días darte cuenta de cuando ocurre estas sensaciones y cómo es tu forma de actuar y lo comentamos la próxima sesión. Un saludo!_
- **R13** patient: _Hola, sí perfecto._
  - therapist (prev): _¡Hasta la próxima!_
- **R14** patient: _Sí, he tenido algunas exposiciones en clase._
  - therapist (prev): _¿Cómo ha ido la semana? ¿Ha ocurrido algo de lo que estuvimos hablando la sesión anterior?_
- **R15** patient: _Pues me he sentido muy nerviosa, lo he pasado mal. ¿A qué te refieres con patrón?_
  - therapist (prev): _¿Cómo te has sentido durante esas exposiciones? ¿Has notado algún patrón en cómo te has relacionado con el malestar que has sentido?_
- **R16** patient: _Ha sido igual que siempre, me pongo nerviosa antes de salir. Me empiezan a sudar las manos, me entra calor y me dan palpitaciones. Durante la exposición se mantiene o aumenta si me equivoco o noto que me tiembla la voz o me quedo en blanco o digo mucho alguna muletilla. Al acabar me tranquilizo, per_
  - therapist (prev): _Me refiero a si has notado que siempre te sientes igual en situaciones como esa, o si ha sido diferente esta vez. Ya sea por como te has desenvuelto durante la exposición o las sensaciones que has notado al acabar._
- **R25** patient: _No me ha pasado hasta ahora, pero me puede pasar y no quiero que esa situación se de._
  - therapist (prev): _Según te dice tu experiencia no ha habido ni una sola vez donde no has sido capaz de bajarte. Aún sintiendo incertidumbre, miedo o ansiedad, has sido capaz de manejar la situación con un 100% de éxito. Eso es muy valioso._
- **R26** patient: _Me gustaría actuar diferente pero al mismo tiempo no, al pensar en la situación me echo para atrás._
  - therapist (prev): _Es comprensible. Hasta ahora has elegido evitar vivir esta situación, pero tú misma me has dicho que estas viendo cada vez menos vínculos que son importantes para ti. ¿Te compensa esta elección o te gustaría actuar de manera diferente_
- **R27** patient: _Sí, me iré a mi casa. Pero viene mi madre en taxi porque me tengo que llevar cosas y no estoy cerca precisamente._
  - therapist (prev): _¿Tienes planeado viajar esta Semana Santa?_
- **R28** patient: _En este caso no. Aunque me gustaría ir algún fin de semana a Córdoba para ver a mis amigas. Solo que al final no me decido nunca._
  - therapist (prev): _¿No hay ningún momento del viaje donde lo hagas tú sola?_
- **R29** patient: _En principio no sabría porque también tendría que saber si ese finde están allí y si puedo quedarme en su piso. Además, hay findes que viene mi novio y no hay fecha fija._
  - therapist (prev): _Entiendo. Eso suena como una oportunidad para tomar un pequeño paso hacia lo que quieres. ¿Hay algún fin de semana que lo tengas más libre para dejarlo fechado y actuar? Aún sabiendo todas estas sensaciones desagradables que pueden aparecer._
- **R30** patient: _Podría intentarlo._
  - therapist (prev): _Bueno, podríamos empezar por planear poco a poco el viaje. Ve contactando con tus amigas y según cómo estén de disponibles podemos fechar el viaje para la próxima sesión._

**Commentary:** _(fill in)_

---

## CompACT-10

### S04 — CompACT-10

- Gold band: `n/a` (total 44)
- Submitted (round 30): total 30, MAE_items=1.400
- Replay (round 17): total 35, MAE_items=0.900
- |ΔMAE| = 0.500

**Items (pred at last available round vs gold):**

| Item | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pred (submitted) | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 30 |
| gold | 5 | 4 | 4 | 4 | 4 | 4 | 5 | 5 | 4 | 5 | 44 |

| Item | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pred (replay) | 3 | 3 | 4 | 3 | 3 | 3 | 4 | 4 | 4 | 4 | 35 |
| gold | 5 | 4 | 4 | 4 | 4 | 4 | 5 | 5 | 4 | 5 | 44 |

**LLM reasoning at submitted round 30 (CoT excerpt):**

_(no CoT logged)_

**LLM reasoning at replay round 17 (CoT excerpt):**

_(no CoT logged)_

**Transcript slice (patient turns; therapist turns where logged):**

- **R13** patient: _Se niega a recibirlo, solo tiene una medicación que le mandaron hace muchísimos años, se limita a tomarse las pastillas y su actividad es nula._
  - therapist (prev): _Por lo que me cuentas, entiendo que por la situación de tu madre quizás has estado muy pendiente de ella. ¿Está recibiendo tratamiento psicológico o psiquiátrico?_
- **R14** patient: _Desde niña intentar constantemente que eso cambiará y que dentro de la complejidad del problema la ayuda pudiese hacer que llevase una vida mejor o que pudiésemos llevarla juntas. Pero cuando todos los intentos fracasaron, me limite a acompañarla en su estado, a veces pasando a ser una sombra de ese_
  - therapist (prev): _¿Cuando vivias con ellos , qué hacías tú en relación a ello? ¿Como es la relación con tu madre en concreto? Disculpa que te haga tantas preguntas, es que necesito entender para ver cómo te puedo ayudar_
- **R15** patient: _Sí, es exactamente la palabra que usaría, resentimiento. Es un sentimiento que ha subido progresivamente durante los años, de modo que a veces encuentro que esos momentos cuando salta todo por los aires son provocados por mí, como si ya no los aguantase, o tuviese algo dentro pudriéndose que me hace_
  - therapist (prev): _Comprendo. Parece una dinámica familiar compleja y agotadora, y es natural que te sientas atrapada entre el cariño y la frustración. Parece que has asumido un rol de cuidadora desde muy pequeña, incluso a costa de tu propio bienestar. ¿Te has sentido resentida en algún momento por ello?_
- **R16** patient: _Sí completamente culpable, porque ese resentimiento suele crear conflictos, y yo no llevo nada bien estar mal con ellos._
  - therapist (prev): _Es muy valiente reconocer eso. El resentimiento es una emoción muy compleja y, a menudo, difícil de expresar. Parece que has estado conteniendo ese sentimiento durante mucho tiempo, y que ahora está saliendo a la superficie de formas que quizás no te gustan. ¿Te has sentido culpable por sentirte así_
- **R17** patient: _No sé si es por la culpa o porque aunque hay una comunicación total, hay una inexistente comunicación emocional, pero no soy capaz de expresarlo con ellos. En cuanto a los pensamientos negativos, no tengo duda de que sí me hace más vulnerable a ellos._
  - therapist (prev): _Es importante reconocer esa culpa, pero también es esencial entender que sentir resentimiento no significa que seas mala persona. Es una emoción legítima y, a veces, necesaria para sanar. ¿Crees que esa culpa te impide expresar lo que sientes o te hace más vulnerable a esos pensamientos negativos?_
- **R25** patient: _No tengo muchas aficiones, cuando estoy sola ver series o leer.  Pero lo que más disfruto, es pasar tiempo con mis amigas o mi pareja, haciendo cualquier cosa._
  - therapist (prev): _Gracias por la información. A veces, nos pasa eso, que no podemos concentrarnos en las tareas y más si conllevan un esfuerzo de atención y concentración. ¿Sueles dedicar tiempo a algo que te guste, hobby?_
- **R26** patient: _Sí eso me despeja, tengo total confianza con ellos, les suelo contar mi día a día y lo que me preocupa, el tema que hablamos ayer del miedo a la muerte o que pasé algo no lo hablo tan a menudo, porque tampoco quiero molestarles demasiado con algo que no tiene solución._
  - therapist (prev): _Me alegro que me digas que te gusta estar con gente cercana a ti como tus amigas y pareja ¿Qué confianza tienes con ellos? ¿les cuentas cómo te sientes? ¿lo que te preocupa?_
- **R27** patient: _No mucho, se limitan a escuchar y a validar lo que siento. Trato de evitar hablarlo últimamente porque me produce miedo que me de un ataque como otras veces si lo pienso de más._
  - therapist (prev): _Entiendo. A veces, puede ser difícil compartir cosas que nos parecen pesadas o que parecen que no tienen solución, pero es cierto que a menudo nos sentimos aliviados al hablar de ellas con alguien que nos importa. ¿Cuando a veces, les has comentado algo al respecto, ¿Cómo han reaccionado? ¿qué te su_
- **R28** patient: _cuando lo pienso demasiado llega un momento en que es como si el miedo se desbordase, es una sensación de terror intenso como si sintiese que me estoy muriendo, dura unos segundos y luego me quedo muy cansada._
  - therapist (prev): _¿cuando lo hablas te dan ataques? dime más sobre esto por favor_
- **R29** patient: _Noto falta de aire y una presión fuerte en el pecho, mareo y un miedo muy muy intenso, más del que allá sentido en cualquier otra ocasión de mi vida.  Los pensamientos son relacionados con el miedo a morir, la primera vez que me pasó tenía 11 años y como no lo conocía tuve la sensación de que estaba_
  - therapist (prev): _Vaya, entiendo que si eso te ha pasado, intentes evitar que suceda de nuevo. Si no te importa, y sabiendo que es algo que no te gusta y como te he dicho, intentas evitar, por favor, descríbeme lo que te pasa. Indícame qué sensaciones físicas tienes (qué notas en cuerpo), qué pensamientos y/o imágene_
- **R30** patient: _Esa sensación solo segundos, es horrible pero muy breve_
  - therapist (prev): _¿Cuanto tiempo te dura?_
- **R31** patient: _Estaba en la cama antes de dormir, fue porque empecé a tener consciencia de la muerte y lo que significaba, estaba pensando en ello._
  - therapist (prev): _Me dices que el primer ataque fue a los 11 años. ¿Recuerdas donde estabas y qué pasó que pudiera ser el desencadentante?_
- **R32** patient: _Sí, siempre es cuando pienso en la muerte_
  - therapist (prev): _sí es cierto, aunque sean segundos, la sensación es muy muy desagradable ¿Te sigue pasando ahora igual, es cuando piensas en la muerte?_
- **R33** patient: _Pienso en la nada que hay después, no soy creyente así que es complicado aceptar y relacionarme con el concepto de la muerte. Pienso en el vacío, en la nada y en mis seres queridos._
  - therapist (prev): _Cuéntame un poco más de los pensamientos concretos que tienes_
- **R34** patient: _En el hecho de que cuando ocurra su muerte o la mía, no les veré más._
  - therapist (prev): _Sobre tus seres queridos qué piensas?_
- **R35** patient: _Para mí se traduce en soledad, soy una persona que valora mucho las relaciones sociales con la gente que verdaderamente me importa. Son un punto clave de mi vida, por eso pienso en cuando falten o en cuando yo muera, aunque lo segundo no debería preocuparme porque ya no existiré, no puedo evitar que_
  - therapist (prev): _Y si no los ves más, eso qué significa para ti?_

**Commentary:** _(fill in)_

---

### S05 — CompACT-10

- Gold band: `n/a` (total 40)
- Submitted (round 30): total 35, MAE_items=1.100
- Replay (round 17): total 37, MAE_items=0.900
- |ΔMAE| = 0.200

**Items (pred at last available round vs gold):**

| Item | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pred (submitted) | 3 | 3 | 3 | 3 | 4 | 4 | 4 | 3 | 4 | 4 | 35 |
| gold | 3 | 1 | 5 | 5 | 4 | 5 | 3 | 4 | 6 | 4 | 40 |

| Item | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pred (replay) | 3 | 3 | 4 | 3 | 4 | 4 | 4 | 4 | 4 | 4 | 37 |
| gold | 3 | 1 | 5 | 5 | 4 | 5 | 3 | 4 | 6 | 4 | 40 |

**LLM reasoning at submitted round 30 (CoT excerpt):**

_(no CoT logged)_

**LLM reasoning at replay round 17 (CoT excerpt):**

- **item 1** (score=3): No direct evidence of rushing through activities.
- **item 2** (score=2): Patient describes disconnection from studies and values.
- **item 3** (score=4): Patient attempts to suppress thoughts.
- **item 4** (score=2): Similar to Item 2, disalignment with values.
- **item 5** (score=3): No clear evidence of situational avoidance.
- **item 6** (score=5): Patient is often inattentive during class.
- **item 7** (score=3): No direct evidence of persistence.
- **item 8** (score=4): Patient attempts to suppress emotions and thoughts.
- **item 9** (score=5): Patient frequently operates on autopilot.
- **item 10** (score=3): No clear evidence of perseverance.

**Transcript slice (patient turns; therapist turns where logged):**

- **R13** patient: _De acuerdo, gracias._
  - therapist (prev): _Bueno, vamos a ver si entre las dos encontramos la manera. Te escribo a lo largo de la mañana para ver cuando podemos retomar la sesión. Perdona, pero tengo otro compromiso ahora mismo._
- **R14** patient: _Hola… regular… hoy he tenido un mal día, en clase me he sentido una completa inútil y que he ido solo a calentar la silla. Intento mantener la atención y escuchar al profesor, pero no soy capaz, cuando llevo un rato me disperso, y acabo haciendo cualquier otra cosa con el ordenador que no sea atende_
  - therapist (prev): _Muchas gracias por contarmelo todo tan bien. Espero que podamos ir avanzando en las próximas semanas_
- **R15** patient: _Normalmente en mis propios pensamientos, incluso entro como en un bucle de quiero atender pero no lo estoy haciendo porque no soy capaz. Me pongo a pensar en los exámenes que tenemos la semana que viene y que todos mis compañeros llevan genial, pero ellos solo viven por y para estudiar, siempre en l_
  - therapist (prev): _Siento oír eso, Lucía. Es un día difícil, y lo que describes suena como un peso bastante grande. Pero vamos a intentar ir despacio. Si yo no lo he entendido mal tú vas a clase y tratas de escuchar al profesor, pero en algún momento te dispersas, no sé si en tus propios pensamientos o te distraer con_
- **R16** patient: _Pues no sé, depende del día... de la asignatura... no me ha dado por mirar el reloj y ver cuánto tiempo pasa, no sabría decirte_
  - therapist (prev): _Entiendo. Es como si tu mente estuviera en una especie de "mira esto, mira esto" constantemente. Y cuando ocurre eso, ¿cuánto tiempo pasa hasta que te das cuenta de que te has desconectado y no estás atendiendo al profesor?_
- **R17** patient: _Creo que cuando llevo un rato, como que de buenas a primeras digo ostras, que me he ido y no  me he enterado de nada de lo que ha dicho... y le acabo preguntando a un compañero cercano, pero me da cosa hacerlo porque no quiero hablar en clase y molestar a mi compañero o distraer al profesor porque a_
  - therapist (prev): _Está bien, no pasa nada. No es necesario saber el tiempo exacto. Pero, ¿dirías que te das cuenta pronto o que solo eres consciente de ello cuándo ya llevas un rato?_
- **R25** patient: _Por eso me preocupa, porque se supone que no me debe pasar, ¿no? Es que es muy raro sentir como todo el mundo se entera de todo y sigue el ritmo y yo voy pasos por detrás siempre..._
  - therapist (prev): _Si te sirve de consuelo, eso me pasa constantemente. Yo lo atribuyo a la edad, pero no creo que ese sea tu caso, jajaja_
- **R26** patient: _Me encantaría trabajarlo, porque de verdad sentir tu mente como una locomotora todo el rato a veces muy muy muy muy muuuuuy rápida y otras muy muy lenta es complicado. Y ahora mismo creo que no sería capaz de analizar mi patrón porque creo que estoy como en modo automático y me cuesta y necesitaría _
  - therapist (prev): _Mira, yo no sé si tienes un problema de atención o no. Si lo necesitas podríamos hacer una valoración más adelante. Pero es poco probable que uno tenga TDAH y unos días sea capaz de estar atenta y otros no. Pero me parece importante que analicemos el patrón, en qué situaciones te distraes más, en cu_
- **R27** patient: _Lo intentaré a ver qué pasa... no sé muy bien como hacerlo... pero no sé... ¿cómo se hace eso?_
  - therapist (prev): _Es comprensible que te preocupara. Y ya tendremos tiempo para hablar sobre ello, pero, como te decía, lo importante ahora es empezar a explorar el patrón e ir cogiendo práctica. ¿Qué te parece si, para la próxima sesión, intentamos hacer un pequeño experimento? Te propongo que durante esta semana, c_
- **R28** patient: _Vale genial así lo he entendido mejor, lo intentaré a ver qué pasa_
  - therapist (prev): _Lo normal cuando uno tiene tendencia a dispersarse es que lo haga con frecuencia y tarde mucho en darse cuenta de que estaba perdida. Y que ademas, cuando se de cuenta, lo deje por imposible. Es como cuando alguien deja de fumar y tiene una recaída. Muchos fumadores cuando se han fumado el primero s_
- **R29** patient: _Hola, pues bueno, intenté hacer lo que me mandaste y es verdad que en el momento me sirvió, pero solo con por ejemplo los juegos de mesa, me volví a ver en esa situación, pero en lugar de entrar en bucle de “que mal no me estoy enterando” conseguí no juzgarme por ello, y me sentí mucho mejor y no pa_
  - therapist (prev): _Perfecto, Lucía. Y en la siguiente sesión ya me cuentas cómo te ha ido. Nos vemos la semana que viene._
- **R30** patient: _Supongo que los juegos de mesa al final es algo que me gusta y forma parte de mi tiempo libre, entonces no siento esa presión. Pero los estudios los veo como una obligación, lo cual también me da coraje, porque ojalá poder verlos como una oportunidad para aprender, pero va todo tan rápido que me que_
  - therapist (prev): _Entiendo que te sientas frustrada, Lucia. Es normal sentirse así cuando se invierte tanto esfuerzo y no se ve el resultado esperado. Suena agotador vivir con esa presión constante. Me parece muy valiente que hayas reconocido y descrito con tanta claridad lo que sucede. Y también, que hayas notado la_
- **R31** patient: _María_
  - therapist (prev): _Es muy interesante lo que dices sobre la diferencia entre el tiempo libre y la obligación. Es natural desear ver los estudios como una oportunidad, pero cada uno los ve como los ve, y por tu experiencia, eso no parece fácil de cambiar. Y no es que tú estés haciendo algo mal, es simplemente que igual_
- **R32** patient: _Mmm... Carmen (?), aunque me sale ponerte María aunque me hayas dicho eso jajaja_
  - therapist (prev): _Ahora te voy a pedir, que te olvides del nombre de tu madre, como si nunca lo hubieras sabido. A partir de ahora tu madre de llama Carmen... ¿Cómo se llama tu madre?_
- **R33** patient: _Intentar instalar un programa nuevo? Aunque el anterior siga estando? Porque borrarse no se puede borrar jajaja. Y sí, parte del problema sería ese..._
  - therapist (prev): _Es la respuesta natural a una larguísima historia. Es como si tuvieras un "botón" que dice "María" y no puedes cambiar eso, aunque quisieras decir Carmen. Es como si tuvieras un programa instalado en tu mente que no se puede actualizar fácilmente. Y lo mismo pasa con los estudios, llevas viéndolos d_
- **R34** patient: _pues no estoy ni estudiando, ni disfrutando ni nada. Mi cabeza se va y se pone a pensar en mil cosas que nada que ver con el estudio_
  - therapist (prev): _Me gusta mucho tu analogía del programa, ¡es muy precisa! Es como si tuvieras dos programas corriendo al mismo tiempo: uno que te dice "esto es obligatorio y difícil" y otro que intenta decirte "esto puede ser interesante". Y ambos están compitiendo por tu atención. Y mientras estás focalizada en es_
- **R35** patient: _Sí_
  - therapist (prev): _¿Qué te parece si hacemos un pequeño ejercicio?, ¿tienes papel a mano?_

**Commentary:** _(fill in)_

---
