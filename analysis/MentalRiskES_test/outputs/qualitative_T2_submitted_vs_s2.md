# Task 2 — Submitted Run 2 vs S2 (Gemma 4 31B + guardrails)

**Comparison cohort:** 300 (round, session) pairs (rounds 1–30 × 10 sessions, the inner join of submitted predictions and S2 predictions on the test set).

**Headline accuracy on this slice**

| System | Accuracy on join | Comment |
| --- | --- | --- |
| Submitted Run 2 (HYB B+) | **0.247** | matches official leaderboard 0.247 |
| Gemma 4 31B + guardrails (S2) | **0.417** | bare LLM + anti-bias guardrails |
| Agreement rate | **0.380** | both systems pick the same option |

**Bucket counts**

| Bucket | Count | Share |
| --- | --- | --- |
| `both_correct` | 34 | 11.3% |
| `s2_wins` | 91 | 30.3% |
| `submitted_wins` | 40 | 13.3% |
| `both_wrong_same` | 80 | 26.7% |
| `both_wrong_disagree` | 55 | 18.3% |

**Per-gold-class accuracy**

| Gold | n | Submitted acc | S2 acc | Agreement |
| --- | --- | --- | --- | --- |
| 1 | 101 | 0.168 | 0.663 | 0.366 |
| 2 | 95 | 0.326 | 0.274 | 0.411 |
| 3 | 104 | 0.250 | 0.308 | 0.365 |

---

## Section A — S2 wins (S2 picks gold, Submitted misses)

_91 cases total; sampling 8 stratified by gold class._

### #1 — S05 round 17  (gold=1, submitted=3, S2=1)

**Recent transcript:**

- **R14 patient (es)**: _Hola… regular… hoy he tenido un mal día, en clase me he sentido una completa inútil y que he ido solo a calentar la silla. Intento mantener la atención y escuchar al profesor, pero no soy capaz, cuando llevo un rato me disperso, y acabo haciendo cualquier otra cosa con el ordenador que no sea atende_
  - **(en)**: _Hello... regular... today I had a bad day, in class I felt completely useless and that I went just to warm the chair. I try to keep my attention and listen to the teacher, but I'm not able to, when I've been there for a while I get scattered, and I end up doing anything else with the computer other _
- **R15 patient (es)**: _Normalmente en mis propios pensamientos, incluso entro como en un bucle de quiero atender pero no lo estoy haciendo porque no soy capaz. Me pongo a pensar en los exámenes que tenemos la semana que viene y que todos mis compañeros llevan genial, pero ellos solo viven por y para estudiar, siempre en l_
  - **(en)**: _Normally in my own thoughts, I even get into a loop of I want to attend but I am not doing it because I am not able to. I start thinking about the exams we have next week and that all my classmates are doing great, but they only live for and to study, always in the library, and I see life in a diffe_
- **R16 patient (es)**: _Pues no sé, depende del día... de la asignatura... no me ha dado por mirar el reloj y ver cuánto tiempo pasa, no sabría decirte_
  - **(en)**: _Well, I don't know, it depends on the day... on the subject... I haven't felt like looking at the clock and seeing how much time goes by, I wouldn't know how to tell you_
- **R17 patient (es)**: _Creo que cuando llevo un rato, como que de buenas a primeras digo ostras, que me he ido y no  me he enterado de nada de lo que ha dicho... y le acabo preguntando a un compañero cercano, pero me da cosa hacerlo porque no quiero hablar en clase y molestar a mi compañero o distraer al profesor porque a_
  - **(en)**: _I think that when I've been there for a while, I kind of just go, oops, I'm gone and I didn't hear anything he said... and I end up asking a close classmate, but I don't want to do it because I don't want to talk in class and bother my classmate or distract the teacher because it was my fault in the_

**Candidate options:**

- **Option 1** — **GOLD** / **S2** (es): _Y entonces, si ya ha pasado un rato y te sientes perdida y no te atreves a preguntarle a un compañero, ¿Qué haces entonces?_
  - **(en)**: _And then, if it's been a while and you feel lost and don't dare ask a colleague, what do you do then?_
- **Option 2** (es): _Entiendo. Es como si te cayera un peso de encima cuando te das cuenta de que te has ido. Pero en lugar de quedarte en eso, ¿qué sucede cuando piensas "he sido una inútil, no he atendido al profesor"? ¿Qué te pasa en ese momento?_
  - **(en)**: _I understand. It's like a weight falls off your shoulders when you realize you've left. But instead of dwelling on that, what happens when you think, "I've been useless, I haven't attended to the teacher"? What happens to you in that moment?_
- **Option 3** — **SUBMITTED** (es): _Eso suena como algo que te cuesta bastante, Lucía. Me imagino que te sientes un poco atrapada entre lo que quieres hacer y lo que haces. ¿Qué te pasa cuando te das cuenta de que no estás atendiendo? ¿Te sientes molesta, frustrada, o quizás algo más?_
  - **(en)**: _That sounds like something you struggle with, Lucia. I imagine you feel a bit caught between what you want to do and what you do. What happens to you when you realize you're not paying attention? Do you feel annoyed, frustrated, or maybe something else?_

**S2 reasoning:** _It is the most natural and direct follow-up to the patient's last statement, maintaining the flow of the narrative to understand the full cycle of her behavior in class._

**Notes:** _(fill in)_

---

### #2 — S07 round 7  (gold=2, submitted=3, S2=2)

**Recent transcript:**

- **R4 patient (es)**: _No me pasa cuanso la interacción es con una persona, pero conforme se van sumando más es cuanso más me preocupa_
  - **(en)**: _It doesn't happen to me when the interaction is with one person, but the more people I interact with, the more I worry about it._
- **R5 patient (es)**: _Sí, en los grupos grandes a penas hablo, ya sea en persona o por whatsapp. Y evito viajar si no es necesario_
  - **(en)**: _Yes, in large groups I hardly talk, either in person or by whatsapp. And I avoid traveling if it is not necessary_
- **R6 patient (es)**: _Me siento mejor cuando me quedo callada o evito viajar, pero luego me siento un poco inútil, poco resolutiva y poco independiente._
  - **(en)**: _I feel better when I keep quiet or avoid traveling, but then I feel a bit useless, unresolute and not very independent._
- **R7 patient (es)**: _Pues en el caso de las exposiciones en clase, por ejemplo, me pongo muy nerviosa porque me voy a sentir observada y juzagada y no quiero hacer el ridículo así que me empiezo a poner nerviosa, me aumentan las pulsaciones, siento calor, me sudan las manos y me tiemblan. Aún así salgo a exponer porque _
  - **(en)**: _In the case of class presentations, for example, I get very nervous because I am going to feel observed and judged and I don't want to make a fool of myself, so I start to get nervous, my heart rate increases, I feel hot, my hands sweat and shake. Even so, I go out to exhibit because I have no other_

**Candidate options:**

- **Option 1** (es): _Vaya, suena a una experiencia bastante intensa. Es normal que te sientas así, con esa activación física y mental. Parece que tu cuerpo está reaccionando como si la exposición en clase fuera una situación de peligro. Me llama la atención que, a pesar de todo eso, sales a exponer. Eso ya demuestra una gran capacidad de resistencia. Y lo de desconectarte de la realidad después, esa sensación de irrealidad, es algo que a veces ocurre cuando la ansiedad es muy alta. Es como si tu mente intentara prot_
  - **(en)**: _Wow, sounds like a pretty intense experience. It's normal for you to feel that way, with that physical and mental activation. It sounds like your body is reacting as if the exposure in class is a dangerous situation. It strikes me that, in spite of all that, you go out to exhibit. That already shows great resilience. And disconnecting from reality afterwards, that feeling of unreality, is something that sometimes happens when anxiety is very high. It is as if your mind is trying to protect you, _
- **Option 2** — **GOLD** / **S2** (es): _Vaya, suena a una experiencia bastante intensa. Me llama la atención que, a pesar de todo eso, sales a exponer. Eso ya demuestra una gran habilidad que va a ser clave en el proceso terapéutico. Y lo de desconectarte de la realidad después, esa sensación de irrealidad, es algo que a veces ocurre cuando la ansiedad es muy alta. Cuando empiezas a notar esas sensaciones de las que me hablas, ¿intentas "luchar" contra ellas? Es decir, ¿haces o piensas cosas para intentar reducir la ansiedad?_
  - **(en)**: _Wow, sounds like a pretty intense experience. It strikes me that, in spite of all that, you come out to expose. That already shows a great skill that is going to be key in the therapeutic process. And the feeling of disconnecting from reality afterwards, that feeling of unreality, is something that sometimes happens when anxiety is very high. When you start to notice those sensations you are talking about, do you try to "fight" against them? That is, do you do or think things to try to reduce th_
- **Option 3** — **SUBMITTED** (es): _Eso suena muy difícil, ¿verdad? Parece que, en esas situaciones, tu cuerpo y mente se ponen en alerta, como si estuvieran preparándose para algo que no quieres enfrentar. Y luego, aunque sales a hablar, hay una especie de desconexión, como si todo fuera una película que se reproduce en segundo plano. ¿Te gustaría que te ayudara a ver eso desde otra perspectiva, como si fuera una historia que estuvieras contando, sin juzgarla?_
  - **(en)**: _That sounds very difficult, doesn't it? It seems like, in those situations, your body and mind go on alert, like they're preparing for something you don't want to face. And then, even though you go out to talk, there's a kind of disconnect, as if everything is a movie playing in the background. Would you like me to help you see that from another perspective, as if it's a story you're telling, without judging it?_

**S2 reasoning:** _It validates the patient's strength and immediately pivots to exploring the 'struggle' (experiential avoidance), which is a core component of the ACT model before jumping into exercises._

**Notes:** _(fill in)_

---

### #3 — S04 round 15  (gold=3, submitted=2, S2=3)

**Recent transcript:**

- **R12 patient (es)**: _Mis padres actualmente viven a 3 horas. Tienen 65 años ambos, mi madre tiene un problema de salud mental crónico, y mi padre tiene una buena salud general, pero me preocupan sus malos hábitos que creo que acabarán por perjudicar ese buen estado, de hecho pienso que es raro que aún no haya ocurrido._
  - **(en)**: _My parents currently live 3 hours away. They are both 65 years old, my mother has a chronic mental health problem, and my father is in good general health, but I am concerned about his bad habits which I think will eventually undermine that good health, in fact I think it is odd that it hasn't happe_
- **R13 patient (es)**: _Se niega a recibirlo, solo tiene una medicación que le mandaron hace muchísimos años, se limita a tomarse las pastillas y su actividad es nula._
  - **(en)**: _He refuses to receive it, he only has a medication that was sent to him many, many years ago, he only takes the pills and his activity is null._
- **R14 patient (es)**: _Desde niña intentar constantemente que eso cambiará y que dentro de la complejidad del problema la ayuda pudiese hacer que llevase una vida mejor o que pudiésemos llevarla juntas. Pero cuando todos los intentos fracasaron, me limite a acompañarla en su estado, a veces pasando a ser una sombra de ese_
  - **(en)**: _Since I was a child, I have constantly tried to change that and that within the complexity of the problem, help could help her lead a better life or that we could lead a better life together. But when all the attempts failed, I limited myself to accompany her in her state, sometimes becoming a shado_
- **R15 patient (es)**: _Sí, es exactamente la palabra que usaría, resentimiento. Es un sentimiento que ha subido progresivamente durante los años, de modo que a veces encuentro que esos momentos cuando salta todo por los aires son provocados por mí, como si ya no los aguantase, o tuviese algo dentro pudriéndose que me hace_
  - **(en)**: _Yes, that's exactly the word I would use, resentment. It's a feeling that has crept up progressively over the years, so that sometimes I find that those moments when it all blows up are triggered by me, like I can't take it anymore, or I have something festering inside that makes me defensive and co_

**Candidate options:**

- **Option 1** (es): _Entiendo que sientes un gran resentimiento hacia la situación y hacia ti misma por haber asumido ese rol. Ese sentimiento de que estás "pudriéndote" interiormente es muy común en situaciones como la tuya. ¿Crees que podrías dejar de lado ese rol de cuidadora y priorizar tu propia necesidad de cuidarte a ti misma?_
  - **(en)**: _I understand that you feel a great deal of resentment towards the situation and yourself for taking on that role. That feeling that you are "rotting" inside is very common in situations like yours. Do you think you could let go of that caregiver role and prioritize your own need to take care of yourself?_
- **Option 2** — **SUBMITTED** (es): _Eso suena como una carga emocional muy intensa, y es comprensible que te sientas así. Parece que el resentimiento ha sido una forma de protegerte, pero también te ha estado consumiendo. ¿Te has preguntado en qué momento empezó a hacerse más evidente ese sentimiento de resentimiento?_
  - **(en)**: _That sounds like a very intense emotional burden, and it's understandable that you feel that way. It sounds like resentment has been a way of protecting you, but it has also been consuming you. Have you wondered at what point that feeling of resentment started to become more evident?_
- **Option 3** — **GOLD** / **S2** (es): _Es muy valiente reconocer eso. El resentimiento es una emoción muy compleja y, a menudo, difícil de expresar. Parece que has estado conteniendo ese sentimiento durante mucho tiempo, y que ahora está saliendo a la superficie de formas que quizás no te gustan. ¿Te has sentido culpable por sentirte así?_
  - **(en)**: _It is very brave to acknowledge that. Resentment is a very complex emotion and often difficult to express. It sounds like you've been holding that feeling in for a long time, and now it's coming to the surface in ways you may not like. Have you felt guilty for feeling that way?_

**S2 reasoning:** _It validates the patient's vulnerability and explores the likely presence of guilt, which aligns with the patient's earlier mention of feeling guilt and the ACT focus on accepting difficult emotions._

**Notes:** _(fill in)_

---

### #4 — S16 round 24  (gold=1, submitted=3, S2=1)

**Recent transcript:**

- **R21 patient (es)**: _Ayer tuve que ir a un cumpleaños donde había gente que no conocía y me empezaron a venir pensamientos sobre que no estaba seguro cayéndole muy bien a la gente. Al principio me agobiaba más cada vez que me daba cuenta de que lo estaba pensando, pero es verdad que conforme avanzó la tarde fui capaz de_
  - **(en)**: _Yesterday I had to go to a birthday party where there were people I didn't know and thoughts started to come to me that I wasn't sure people liked me very much. At first I was more overwhelmed every time I realized I was thinking it, but it is true that as the evening progressed I was able to notice_
- **R22 patient (es)**: _No, es verdad que siempre he sabido reconocer que tengo un problema con la validación masculina… Pero ayer pude darme aún más cuenta. Es algo con lo que no me siento nada agusto, y me da mucha rabia que me pase. Desde adolescente tengo esa sensación y es algo que no se me va con los años En el insti_
  - **(en)**: _No, it's true that I've always been able to recognize that I have a problem with male validation... But yesterday I was able to realize it even more. It's something I don't feel comfortable with at all, and it makes me very angry that it happens to me. Since I was a teenager I have this feeling and _
- **R23 patient (es)**: _Sí, es justo eso. No me gusta estar pensando eso en vez de disfrutar del cumpleaños de mi amigo y prestarle atención a él, que es para lo que estoy allí_
  - **(en)**: _Yes, that's just it. I don't like to be thinking that instead of enjoying my friend's birthday and paying attention to him, which is what I'm there for._
- **R24 patient (es)**: _La verdad es que sí. Pero era muy corto el tiempo que era capaz de hacerlo, enseguida volvían otra vez…_
  - **(en)**: _The truth is, yes. But it was very short the time I was able to do it, they would come back again..._

**Candidate options:**

- **Option 1** — **GOLD** / **S2** (es): _Entiendo. En realidad esto es un proceso,  y necesitarás mucha más práctica. Pero por algo se empieza. Esto es como si empezáramos a hacer ejercicio por primera vez en muchos años. Probablemente el primer día no tendremos mucho aguante, pero esa es la manera de empezar Y dime, qué pensamientos concretos venían más cuando estabas en esa situación?_
  - **(en)**: _I understand. This is really a process, and you'll need a lot more practice. But it starts somewhere. This is like starting to exercise for the first time in many years. Probably the first day we won't have much stamina, but that's the way to start. And tell me, what specific thoughts came to mind the most when you were in that situation?_
- **Option 2** (es): _Entiendo. Así que todavía es un proceso, pero ya has empezado a notar esos momentos en los que puedes estar más presente y menos en la carpeta de los pensamientos. ¿Te parece que lo que ayer te ayudó a estar más presente fue el hecho de reconocer que estabas pensando en la validación masculina?_
  - **(en)**: _I understand. So it's still a process, but you've already started to notice those moments where you can be more present and less in the folder of thoughts. Do you find that what helped you be more present yesterday was recognizing that you were thinking about male validation?_
- **Option 3** — **SUBMITTED** (es): _Es un proceso, Raquel. No esperes que desaparezca de la noche a la mañana. Lo importante es que te has dado cuenta de que puedes notar la carpeta, que puedes elegir no engancharte a ella, aunque sea por un instante. ¿Qué tal si la próxima vez que notes que se te abre la carpeta, en lugar de juzgarte, te recuerdas a ti misma que estás aprendiendo a manejarla?_
  - **(en)**: _It's a process, Raquel. Don't expect it to disappear overnight. What's important is that you've realized that you can notice the folder, that you can choose not to get hooked on it, even if only for a moment. How about the next time you notice the folder opening up for you, instead of judging yourself, remind yourself that you're learning how to handle it?_

**S2 reasoning:** _It validates the patient's struggle with the 'short duration' of presence using a helpful analogy and then pivots back to the specific content of the thoughts to deepen the clinical work._

**Notes:** _(fill in)_

---

### #5 — S12 round 4  (gold=2, submitted=3, S2=2)

**Recent transcript:**

- **R1 patient (es)**: _Hola, buenas tardes_
  - **(en)**: _Hello, good afternoon_
- **R2 patient (es)**: _Bueno, la verdad que he tenido días peores, pero tampoco es que esté del todo bien_
  - **(en)**: _Well, the truth is that I've had worse days, but it's not that I'm not completely well either._
- **R3 patient (es)**: _No he conseguido dormir del todo bien esta noche, encima esta mañana tenía unas tareas pendientes y me han surgido unos imprevistos que me han hecho atrasarlas... Me frustra el no poder haber terminado lo que tenía pensado para hoy_
  - **(en)**: _I have not been able to sleep well tonight, on top of that I had some pending tasks this morning and some unforeseen events have made me delay them.... I'm frustrated that I couldn't finish what I had planned for today._
- **R4 patient (es)**: _me genera incomodidad, no puedo estar quieta y necesito moverme de un lado para el otro todo el tiempo. Aparte siento que se me van a acumular las cosas que tengo que hacer y eso me pone peor_
  - **(en)**: _It makes me uncomfortable, I can't stand still and I need to move from one side to the other all the time. Besides, I feel that the things I have to do are going to pile up and that makes me feel worse._

**Candidate options:**

- **Option 1** (es): _La incomodidad y la necesidad de moverte son respuestas físicas a esa frustración, parece. Y esa sensación de acumulación de tareas... ¿Podrías intentar identificar qué es lo que te preocupa más de que se acumulen esas tareas? ¿Es el tiempo, la dificultad, o algo más?_
  - **(en)**: _Discomfort and the need to move are physical responses to that frustration, it seems. And that feeling of accumulating tasks? Could you try to identify what it is that worries you most about those tasks piling up? Is it the time, the difficulty, or something else?_
- **Option 2** — **GOLD** / **S2** (es): _La incomodidad y la necesidad de moverte son respuestas físicas a esa frustración, parece. Es normal sentirlo. Y esa sensación de acumulación de tareas... ¿Podrías intentar identificar qué es lo que te preocupa más de que se acumulen esas tareas? ¿Es el tiempo, la dificultad, o algo más?_
  - **(en)**: _Discomfort and the need to move are physical responses to that frustration, it seems. It's normal to feel it. And that feeling of accumulating tasks? Could you try to identify what it is that worries you most about those tasks piling up? Is it the time, the difficulty, or something else?_
- **Option 3** — **SUBMITTED** (es): _Entiendo que la incomodidad y la necesidad de moverte son síntomas de un nivel de ansiedad o estrés. Y la sensación de que las tareas se acumulan y te pesan también es un tema importante. ¿Te parece que hay algo en particular que te haga sentir que no puedes controlar la situación o que te sientes abrumada por las responsabilidades?_
  - **(en)**: _I understand that discomfort and the need to move around are symptoms of a level of anxiety or stress. And the feeling that tasks are piling up and weighing you down is also a major issue. Do you find that there is anything in particular that makes you feel like you can't control the situation or that you feel overwhelmed by the responsibilities?_

**S2 reasoning:** _It provides a brief, necessary moment of validation ('Es normal sentirlo') before moving into the functional exploration of the patient's anxiety._

**Notes:** _(fill in)_

---

### #6 — S06 round 10  (gold=3, submitted=2, S2=3)

**Recent transcript:**

- **R7 patient (es)**: _La verdad es que no lo sé. Más allá de mi propia experiencia interna. Esto es de perogrullo, pero supongo que si pudiera disfrutar más de las cosas, pues disfrutaría más y no estaría amargado ¿Cómo cambiaría eso mi vida? Pues no lo sé, igual no lo haría_
  - **(en)**: _The truth is that I don't know. Beyond my own internal experience. This is a truism, but I suppose that if I could enjoy things more, I would enjoy more and I would not be bitter. Well, I don't know, maybe I wouldn't._
- **R8 patient (es)**: _Dinero No te digo ser rico, pero sí dejar de ser pobre_
  - **(en)**: _Money I'm not telling you to be rich, but I'm telling you to stop being poor._
- **R9 patient (es)**: _Si tuviera dinero como para no tener que trabajar y estudiar a la vez, mi vida sería más sencilla y estaría menos cansado. Supongo que eso me ayudaría a afrontar mejor las cosas Por ejemplo, hoy me acosté a las 2 am, que es cuando llegué a mi casa del trabajo. A las 8 am tengo clases y llego a casa _
  - **(en)**: _If I had money so that I didn't have to work and study at the same time, my life would be simpler and I would be less tired. I guess that would help me cope better For example, today I went to bed at 2 am, which is when I got home from work. At 8 am I have classes and I get home at 3:30 pm. At 6:00 _
- **R10 patient (es)**: _Sí, si a mí el dinero como tal me la pela. No es como que lo desee. Ni es el dinero en sí lo que necesito, sino el tiempo de ocio Supongo que es un medio Pero claro, es que la sociedad funciona así. Yo por mí, pues no trabajaría, pero entonces no podría pagar la casa ni la comida, ni nada No es como_
  - **(en)**: _Yes, I don't care about money as such. It's not like I want it. It's not money per se that I need, but leisure time. I suppose it's a means. I wouldn't work for me, but then I wouldn't be able to pay for the house or the food, or anything. It's not like I want the money, the landlord wants it, the g_

**Candidate options:**

- **Option 1** (es): _Entiendo que el dinero es un medio para algo que realmente te importa: tener tiempo para ti. ¿Qué te gustaría hacer con ese tiempo, aunque sea algo pequeño? ¿Qué sería una forma de "dormir más", o "comer mejor", o "hacer algo que te guste" sin que sea un esfuerzo?_
  - **(en)**: _I understand that money is a means to something that really matters to you: having time for yourself. What would you like to do with that time, even if it's something small? What would be a way to "sleep more", or "eat better", or "do something you like" without it being an effort?_
- **Option 2** — **SUBMITTED** (es): _Tienes razón, Manuel. El dinero es una herramienta para cubrir necesidades básicas y permitirnos hacer cosas que valoramos. Y parece que lo que realmente valoras es el tiempo y la energía para ti mismo. ¿Qué pequeño paso podrías dar esta semana para recuperar un poco de ese tiempo y energía, aunque sea por unos minutos al día?_
  - **(en)**: _You are right, Manuel. Money is a tool to cover basic needs and allow us to do things we value. And it sounds like what you really value is time and energy for yourself. What small step could you take this week to reclaim some of that time and energy, even if it's just for a few minutes a day?_
- **Option 3** — **GOLD** / **S2** (es): _Entiendo que el dinero es un medio para algo que realmente te importa: tener tiempo para ti. ¿Qué te gustaría hacer con ese tiempo, aunque sea algo pequeño?_
  - **(en)**: _I understand that money is a means to something that really matters to you: having time for yourself. What would you like to do with that time, even if it's something small?_

**S2 reasoning:** _It validates the patient's distinction between money and time and asks an open-ended question about values without jumping too quickly into problem-solving or 'small steps' while the patient is feeling exhausted._

**Notes:** _(fill in)_

---

### #7 — S05 round 25  (gold=1, submitted=3, S2=1)

**Recent transcript:**

- **R22 patient (es)**: _No, en sesión no, al menos de momento no he notado esa sensación_
  - **(en)**: _No, not in session, at least so far I haven't noticed that feeling._
- **R23 patient (es)**: _Sinceramente no lo sé, también creo que es algo que depende dl día que me pille. Hay veces que consigo prestar toda mi atención en algo pero hay otros días que me cuesta mucho, me disperso o me quedo en blanco y tengo que pedir que me repitan las cosas y eso_
  - **(en)**: _I honestly don't know, I also think that it depends on the day I get caught. There are times when I manage to pay full attention to something but there are other days that I have a hard time, I get scattered or I go blank and I have to ask for things to be repeated and that's what I do._
- **R24 patient (es)**: _Sí, el río depende del día o de la racha que lleve va muy muy rápido o muy muy lento. Lo que pasa que últimamente siento como que el rio no va a la velocidad que normalmente solía ir. Otro ejemplo que te quería poner, es que lo mismo hay veces que estoy diciendo algo, y de buenas a primeras se me va_
  - **(en)**: _Yes, the river goes either very fast or very slow, depending on the day or the run it is on. The thing is that lately I feel like the river is not going at the speed it normally used to go. Another example I wanted to give you, is that there are times when I am saying something, and all of a sudden _
- **R25 patient (es)**: _Por eso me preocupa, porque se supone que no me debe pasar, ¿no? Es que es muy raro sentir como todo el mundo se entera de todo y sigue el ritmo y yo voy pasos por detrás siempre..._
  - **(en)**: _That's why I'm worried, because it's not supposed to happen to me, right? It's just that it's very strange to feel how everybody knows everything and keeps up the pace and I'm always steps behind?_

**Candidate options:**

- **Option 1** — **GOLD** / **S2** (es): _Mira, yo no sé si tienes un problema de atención o no. Si lo necesitas podríamos hacer una valoración más adelante. Pero es poco probable que uno tenga TDAH y unos días sea capaz de estar atenta y otros no. Pero me parece importante que analicemos el patrón, en qué situaciones te distraes más, en cuáles menos, a qué pensamientos te quedas más enganchada y, si de alguna manera lo que haces, te está ayudando a volver a conectar o no. Soy consciente de que tu mente a veces va muy rápido, porque lo _
  - **(en)**: _Look, I don't know if you have an attention problem or not. If you need it we could do an assessment later. But it is unlikely that you have ADHD and some days you are able to be attentive and others you are not. But it seems important to me that we analyze the pattern, in which situations you get more distracted, in which less, which thoughts you get more hooked on and if somehow what you are doing is helping you to reconnect or not. I am aware that your mind sometimes goes very fast, because I_
- **Option 2** (es): _Y es cierto, a veces nos sentimos así, como si estuviéramos en una carrera, y todo el mundo corre y tú estás un poco más atrás. Pero, ¿qué te gustaría que fuera de esa carrera? ¿Qué sería una vida más valiosa para ti en este momento?_
  - **(en)**: _And it's true, sometimes we feel like that, like we're in a race, and everybody's running and you're a little bit behind. But what would you like out of that race? What would be a more valuable life for you right now?_
- **Option 3** — **SUBMITTED** (es): _Parece que te sientes como si estuvieras en un ritmo diferente al de los demás, y eso te hace sentir un poco fuera de lugar. ¿Te acuerdas de la metáfora del río? ¿Te gustaría explorar un poco más cómo te sientes cuando te sientes "fuera de lugar"?_
  - **(en)**: _You seem to feel like you're in a different rhythm than everyone else, and that makes you feel a little out of place. Remember the metaphor of the river? Would you like to explore a little more how you feel when you feel "out of place"?_

**S2 reasoning:** _The patient is expressing significant anxiety and a need for diagnostic clarity regarding TDAH; Option 1 addresses this concern directly while normalizing the experience and proposing a concrete therapeutic plan._

**Notes:** _(fill in)_

---

### #8 — S04 round 23  (gold=2, submitted=3, S2=2)

**Recent transcript:**

- **R20 patient (es)**: _No, en absoluto, es completo terror a que ocurra, así que jamás la buscaría por mi cuenta._
  - **(en)**: _No, not at all, it's complete terror of it happening, so I would never look for it on my own._
- **R21 patient (es)**: _Vale perfecto, muchas gracias._
  - **(en)**: _Ok perfect, thank you very much._
- **R22 patient (es)**: _Buenos días, estoy bien_
  - **(en)**: _Good morning, I am well_
- **R23 patient (es)**: _Me sentí un poco revuelta al hablar de temas que no suelo tocar mucho._
  - **(en)**: _I felt a little revolted talking about topics that I don't usually touch on much._

**Candidate options:**

- **Option 1** (es): _Es normal sentirse así al abordar temas que no suelo tocar. ¿Qué te gustaría explorar hoy?_
  - **(en)**: _It's normal to feel this way when approaching topics I don't usually touch. What would you like to explore today?_
- **Option 2** — **GOLD** / **S2** (es): _Entiendo. Es natural sentirse así cuando se habla de aquellas cosas que nos preocupan y nos provocan malestar y temor. Ya sabes que puedes ir contándome lo que tú veas. Mi objetivo es, como te dije ayer, es conocerte , conocer lo que te pasa para poder ayudarte. Ayer lo hiciste muy bien, al ir dándome la información que te iba solicitando y te lo agradezco mucho Me gustaría que me contaras sobre lo que sueles hacer en tu día a día, cual es tu rutina_
  - **(en)**: _I understand. It is natural to feel this way when talking about those things that worry us and cause us discomfort and fear. You know that you can tell me what you see. My goal is, as I told you yesterday, is to get to know you, to know what is happening to you so I can help you. Yesterday you did very well, by giving me the information I was asking for and I thank you very much. I would like you to tell me what you usually do in your daily life, what is your routine._
- **Option 3** — **SUBMITTED** (es): _Entiendo. Es natural sentirse así cuando se exploran áreas sensibles. ¿Podrías contarme un poco más sobre qué aspectos te hicieron sentir revuelta? No es necesario que entres en detalles si no te sientes cómoda, pero saber qué te perturbó me ayudaría a comprender mejor tu experiencia._
  - **(en)**: _I understand. It's natural to feel this way when exploring sensitive areas. Could you tell me a little more about what aspects made you feel revolted? You don't need to go into detail if you don't feel comfortable, but knowing what disturbed you would help me better understand your experience._

**S2 reasoning:** _It provides necessary validation and positive reinforcement for the patient's vulnerability while gently transitioning to a more grounded, behavioral exploration of her daily routine._

**Notes:** _(fill in)_

---

## Section B — Submitted wins (Submitted picks gold, S2 misses)

_40 cases total; sampling 8 stratified by gold class._

### #1 — S01 round 24  (gold=1, submitted=1, S2=2)

**Recent transcript:**

- **R21 patient (es)**: _Pues quizás ese factor de desconfianza o evitar ser vulnerable Pero entonces debería ser vulnerable? Debería de dar una confianza que me puede romper en mil pedazos? Eso es muy peligroso, no sabes nunca quién te va a hacer daño_
  - **(en)**: _So maybe that factor of mistrust or avoiding being vulnerable But then should I be vulnerable? Should I give a confidence that can break me into a thousand pieces? That's very dangerous, you never know who is going to hurt you._
- **R22 patient (es)**: _Pues porque dicen que en psicología no se juzga, aunque por dentro no sé que estarás pensando. Y si sigo así, no duraré mucho, demasiados años llevo ya_
  - **(en)**: _Well, because they say that in psychology you don't judge, although inside I don't know what you're thinking. And if I go on like this, I won't last long, I've been there for too many years already._
- **R23 patient (es)**: _Pues quizás me iría mejor, pero también podria sufrir mucho más Y no me gustaría para nada_
  - **(en)**: _Well, maybe I would do better, but I could also suffer much more and I would not like it at all._
- **R24 patient (es)**: _Mi ausencia no significaría mucho, pero no creo que viviría muchos años mas Y no de la mejor manera_
  - **(en)**: _My absence wouldn't mean much, but I don't think I would live many more years, and not in the best way._

**Candidate options:**

- **Option 1** — **GOLD** / **SUBMITTED** (es): _Aquí me estás señalando que vivir así supondría mucho sufrimiento y según me comentas no vivir muchos años con una vida plena. Me parece que es un tema muy importante que me gustaría seguir tratando contigo. ¿Te parece si seguimos abordando este punto con mayor profundidad en la próxima sesión?_
  - **(en)**: _Here you are pointing out to me that living this way would mean a lot of suffering and, according to what you tell me, not living many years with a full life. It seems to me that this is a very important topic that I would like to continue discussing with you, do you think we can continue to address this point in greater depth in the next session?_
- **Option 2** — **S2** (es): _Parece que te has estado cargando con mucho peso durante mucho tiempo. Es comprensible que te sientas agotado y que te preocupe el futuro. ¿Qué pasaría si pudiéramos explorar juntos algunas estrategias para afrontar este peso, sin necesidad de exponerte a todo el dolor que temes?_
  - **(en)**: _It sounds like you've been carrying a lot of weight for a long time. It's understandable that you're feeling drained and worried about the future. What if we could explore together some strategies for dealing with this weight, without exposing you to all the pain you fear?_
- **Option 3** (es): _¿Y si probaras a hacer algo pequeño, aunque sea algo sencillo, que te acerque un poco más a esa vida que mencionas? ¿Qué te gustaría intentar primero?_
  - **(en)**: _What if you tried doing something small, even something simple, that would bring you a little closer to that life you mention? What would you like to try first?_

**S2 reasoning:** _The patient is expressing deep exhaustion and fear of vulnerability; Option 2 provides the necessary validation and a safe bridge toward ACT's focus on psychological flexibility without pushing for immediate action (Option 3) or prematurely ending the topic (Option 1)._

**Notes:** _(fill in)_

---

### #2 — S03 round 18  (gold=2, submitted=2, S2=1)

**Recent transcript:**

- **R15 patient (es)**: _Sii por eso no sabia que hacer, y he acudido a ti_
  - **(en)**: _Yes, that's why I didn't know what to do, and I came to you._
- **R16 patient (es)**: _Si, me gustaria saber que tengo que hacer para volver a mi vida Aunque me va a costar sentirme mal_
  - **(en)**: _Yes, I would like to know what I have to do to get back to my life, although it will be hard for me to feel bad._
- **R17 patient (es)**: _Pues muy mal, no he encontrado ninguna solucion Estoy perdiendo_
  - **(en)**: _Well, too bad, I haven't found any solution._
- **R18 patient (es)**: _No, me siento igual de mal_
  - **(en)**: _No, I feel just as bad_

**Candidate options:**

- **Option 1** — **S2** (es): _Entiendo. Parece que estás en una situación en la que estás intentando evitar el malestar, pero no estás consiguiendo evitarlo. ¿Te parece que hay algo que podrías intentar de manera diferente? Algo que te permita seguir adelante en la dirección que te importa, incluso si eso significa sentirte mal?_
  - **(en)**: _I understand. It sounds like you're in a situation where you're trying to avoid discomfort, but you're not managing to avoid it. Does it sound like there's something you could try differently? Something that would allow you to move forward in the direction that matters to you, even if it means feeling bad?_
- **Option 2** — **GOLD** / **SUBMITTED** (es): _Me has dicho que estás perdiendo, pero me da la impresión de que de algún modo estás metido en un juego en el que es muy difícil ganar. Apuestas todo a intentar controlar el malestar, pero sigue ahí. ¿Se te ocurre otra cosa a la que apostar? Algo en lo que poner tu energía, tu esfuerzo?_
  - **(en)**: _You've told me you're losing, but I get the impression that somehow you're in a game where it's very hard to win. You're betting everything on trying to control the discomfort, but it's still there. Can you think of something else to bet on? Something to put your energy, your effort into?_
- **Option 3** (es): _Entonces, ¿qué te gustaría que fuera diferente? ¿Qué te gustaría que sucediera con tu vida, con tus sentimientos, con el malestar? No necesitas resolverlo todo, solo un pequeño paso hacia algo que te importe. ¿Qué sería ese paso?_
  - **(en)**: _So what would you like to be different? What would you like to happen to your life, to your feelings, to the discomfort? You don't need to solve everything, just a small step towards something that matters to you. What would that step be?_

**S2 reasoning:** _It directly addresses the patient's current struggle with experiential avoidance and introduces the ACT core concept of moving toward values despite the presence of pain._

**Notes:** _(fill in)_

---

### #3 — S05 round 13  (gold=3, submitted=3, S2=2)

**Recent transcript:**

- **R10 patient (es)**: _A veces he intentado dejar el móvil apartado y ponerme a hacer esquemas, ya que escribir me ayuda a concentrarme un poco más que el hecho de solo leer, pero dura poco. Porque cuando me doy cuenta he perdido mucho tiempo intentando hacer el esquema y no siento que haya aprovechado ese tiempo ya que m_
  - **(en)**: _Sometimes I have tried to leave my cell phone aside and start making outlines, since writing helps me to concentrate a little more than just reading, but it doesn't last long. Because when I realize I have lost a lot of time trying to do the outline and I don't feel that I have taken advantage of th_
- **R11 patient (es)**: _creo que me pasa porque reviso también muchas veces que esté bien lo que he puesto en el esquema, si me he saltado información, si tengo todos los apuntes correctos... y pierdo más tiempo en revisar que todo esté bien que en estudiarlo. Lo que suele ocurrir después es que como he estado revisando qu_
  - **(en)**: _I think it happens to me because I also check many times that what I have put in the outline is correct, if I have skipped information, if I have all the correct notes... and I waste more time in checking that everything is correct than in studying it. What usually happens later is that as I have be_
- **R12 patient (es)**: _Pues si me ayudas a encontrarla te lo agradezco, porque todavía no la he encontrado y no sé cómo hacerlo Es que siento que vivo en automático, y en eso los estudios seguramente tengan mucho que ver_
  - **(en)**: _Well, if you help me find it, I'd appreciate it, because I still haven't found it and I don't know how to do it. I feel like I live in automatic mode, and studies probably have a lot to do with that._
- **R13 patient (es)**: _De acuerdo, gracias._
  - **(en)**: _Okay, thank you._

**Candidate options:**

- **Option 1** (es): _De nada, Lucia. Que tengas un buen día._
  - **(en)**: _You are welcome, Lucia. Have a nice day._
- **Option 2** — **S2** (es): _¿Te gustaría que te haga una pregunta diferente? ¿Qué te gustaría que fuera diferente en tu vida?_
  - **(en)**: _Would you like me to ask you a different question? What would you like to be different in your life?_
- **Option 3** — **GOLD** / **SUBMITTED** (es): _Muchas gracias por contarmelo todo tan bien. Espero que podamos ir avanzando en las próximas semanas_
  - **(en)**: _Thank you very much for telling me everything so well. I hope we can make progress in the coming weeks._

**S2 reasoning:** _The patient has just agreed to work on finding their values/direction, and this question is a classic ACT intervention to move from problem-description to value-identification._

**Notes:** _(fill in)_

---

### #4 — S16 round 27  (gold=1, submitted=1, S2=3)

**Recent transcript:**

- **R24 patient (es)**: _La verdad es que sí. Pero era muy corto el tiempo que era capaz de hacerlo, enseguida volvían otra vez…_
  - **(en)**: _The truth is, yes. But it was very short the time I was able to do it, they would come back again..._
- **R25 patient (es)**: _Pues el principal era “no les estás gustando” o “no les estás cayendo tan bien como tus amigas”, porque a veces notaba que con sus comentarios se reían más o le hacían más caso_
  - **(en)**: _Well, the main one was "they are not liking you" or "they are not liking you as much as your friends", because sometimes she noticed that with her comments they laughed more or paid more attention to her._
- **R26 patient (es)**: _Sí, eran nervios sobre todo y agobio, o incluso me sentía ridícula en ocasiones y algo pesada cuando veía que hablaba y no había la reacción que pensaba_
  - **(en)**: _Yes, I was mostly nervous and overwhelmed, or even felt ridiculous at times and a bit heavy when I saw that I was talking and there wasn't the reaction I thought there would be._
- **R27 patient (es)**: _Pues  al principio intentar seguir diciendo cosas hasta conseguir esa atención, y si veía que no funcionaba dejar de hablar durante un rato por lo que te he dicho de sentirme un poco ridícula y pesada También darle vueltas a las cosas que había dicho, a ver si eran tonterías_
  - **(en)**: _Well, at first I tried to keep saying things until I got that attention, and if I saw that it didn't work, I stopped talking for a while because of what I told you about feeling a little ridiculous and heavy._

**Candidate options:**

- **Option 1** — **GOLD** / **SUBMITTED** (es): _Y cuando seguías la conversación o cuando finalmente dejabas de hablar. ¿Qué pasaba con los pensamientos de "no les estás gustando" o con los sentimientos de agobio? ¿Seguían ahí o desaparecían aunque fuera por un momento?_
  - **(en)**: _And when you continued the conversation or when you finally stopped talking, what happened to the thoughts of "they're not liking you" or the feelings of overwhelm? Were they still there or did they disappear even for a moment?_
- **Option 2** (es): _Entiendo. Así que cuando no conseguías la atención que querías, intentabas seguir intentándolo, y si no funcionaba, te sentías ridícula y pesada, y luego empezabas a darle vueltas a las cosas que habías dicho, para ver si eran tonterías. ¿Te parece que hay algo de eso que se repita en la forma en que te relacionas con tus amigos?_
  - **(en)**: _I understand. So when you didn't get the attention you wanted, you'd try to keep trying, and if it didn't work, you'd feel ridiculous and heavy, and then you'd start spinning things you'd said, to see if they were nonsense. Do you find any of that repeating itself in the way you relate to your friends?_
- **Option 3** — **S2** (es): _Así que intentabas "compensar" esa sensación de no ser gustada, buscando la atención de los demás, y cuando no la obtenías, te retirabas y analizabas tus palabras. ¿Te parece que esa estrategia funciona a largo plazo? ¿Te acerca a conectar con la gente o te aleja?_
  - **(en)**: _So you would try to "compensate" for that feeling of not being liked, seeking attention from others, and when you didn't get it, you would withdraw and analyze your words. Do you find that this strategy works in the long run? Does it bring you closer to connecting with people or does it push you away?_

**S2 reasoning:** _This response aligns best with ACT by helping the patient identify the 'workability' of her avoidance/compensation strategies and linking them to her core value of connecting with others._

**Notes:** _(fill in)_

---

### #5 — S03 round 27  (gold=2, submitted=2, S2=1)

**Recent transcript:**

- **R24 patient (es)**: _Buenas tardes miguel_
  - **(en)**: _Good afternoon miguel_
- **R25 patient (es)**: _Buenas tardes_
  - **(en)**: _Good afternoon_
- **R26 patient (es)**: _Buenas_
  - **(en)**: _Good_
- **R27 patient (es)**: _No te preocupes, pues la verdad que mas o menos igual, volvi al trabajo, te acuerdas que te dije que estaba de vacaciones Pero tengo mucha ansiedad en el, y al salir a la calle, aparte me encuentro muy cansado_
  - **(en)**: _Don't worry, the truth is more or less the same, I went back to work, remember I told you I was on vacation, but I have a lot of anxiety in it, and when I go out to the street, I am very tired._

**Candidate options:**

- **Option 1** — **S2** (es): _Sí, me acuerdo de que mencionaste que estabas de vacaciones. Entiendo que la vuelta al trabajo puede ser un poco difícil, sobre todo cuando estás cansado. La ansiedad también es un tema que podemos trabajar juntos. ¿Te parece que la ansiedad te está afectando mucho en tu vida cotidiana, o es más un sentimiento que te queda después de un día de trabajo?_
  - **(en)**: _Yes, I remember you mentioning that you were on vacation. I understand that returning to work can be a bit difficult, especially when you are tired. Anxiety is also an issue that we can work on together. Do you find that anxiety is affecting you a lot in your daily life, or is it more of a feeling that you are left with after a day at work?_
- **Option 2** — **GOLD** / **SUBMITTED** (es): _Sí, claro que me acuerdo. ¿Cómo describirías esa ansiedad?_
  - **(en)**: _Yes, of course I remember. How would you describe that anxiety?_
- **Option 3** (es): _Entiendo. Volver al trabajo después de unas vacaciones puede ser un golpe, y es completamente normal que te sientas así. La ansiedad y el cansancio son respuestas comprensibles ante un cambio de rutina y, posiblemente, ante las demandas del trabajo. Cuéntame un poco más sobre esa ansiedad que sientes en el trabajo. ¿Hay situaciones específicas que la desencadenan? ¿O es una sensación más generalizada? Y cuando dices que te sientes cansado al salir a la calle, ¿cómo describirías ese cansancio? ¿E_
  - **(en)**: _I understand. Returning to work after a vacation can be a shock, and it's completely normal to feel that way. Anxiety and fatigue are understandable responses to a change in routine and, possibly, to the demands of the job. Tell me a little more about that anxiety you feel at work - are there specific situations that trigger it, or is it a more generalized feeling? And when you say you feel tired when you go out on the street, how would you describe that tiredness? Is it physical, mental, or bot_

**S2 reasoning:** _It balances validation of the patient's current state with a focused question to assess the impact of the anxiety, without being overly clinical or overwhelming like Option 3._

**Notes:** _(fill in)_

---

### #6 — S09 round 21  (gold=3, submitted=3, S2=1)

**Recent transcript:**

- **R18 patient (es)**: _Pues sentiría que estoy avanzando si pudiese disfrutar de las situaciones y empezar a hacer más cosas y no siempre las mismas cuatro cosas que siempre hago y que son muy limitadas._
  - **(en)**: _Well, I would feel that I am moving forward if I could enjoy the situations and start doing more things and not always the same four things that I always do and that are very limited._
- **R19 patient (es)**: _Pues quizás me gustaría poder coger un autobús a granada para ver a mi novio aunque creo que es demasiado precipitado un trayecto tan largo_
  - **(en)**: _Well, maybe I would like to take a bus to granada to see my boyfriend, although I think it is too rushed for such a long trip._
- **R20 patient (es)**: _Quizás podría coger el autobús para la universidad y no ir siempre andando por miedo_
  - **(en)**: _Maybe I could take the bus to the university and not walk all the time out of fear._
- **R21 patient (es)**: _Sí la verdad siento que me ayudaría a recuperar un poco mi vida_
  - **(en)**: _Yes, I really feel it would help me to get my life back a little bit._

**Candidate options:**

- **Option 1** — **S2** (es): _Bien. Entonces, ¿qué pensamientos o preocupaciones te surgen cuando piensas en coger el autobús para ir a la universidad?_
  - **(en)**: _Right. So, what thoughts or concerns come to mind when you think about taking the bus to college?_
- **Option 2** (es): _Perfecto, entonces vamos a trabajar en ese objetivo. ¿Cuándo crees que podrías intentar coger el autobús a la universidad? ¿Te gustaría que planeara algo concreto para ayudarte a prepararte para ese primer viaje?_
  - **(en)**: _Perfect, then let's work on that goal. When do you think you might try to catch the bus to college? Would you like me to plan something concrete to help you prepare for that first trip?_
- **Option 3** — **GOLD** / **SUBMITTED** (es): _Perfecto, entonces vamos a trabajar en ese objetivo. ¿Qué pensamientos o preocupaciones te surgen cuando piensas en coger el autobús para ir a la universidad?_
  - **(en)**: _Perfect, then let's work on that goal. What thoughts or concerns come to mind when you think about taking the bus to college?_

**S2 reasoning:** _It transitions naturally from the patient's goal to the ACT process of identifying the internal barriers (thoughts/feelings) without rushing into a behavioral plan too quickly._

**Notes:** _(fill in)_

---

### #7 — S07 round 2  (gold=1, submitted=1, S2=3)

**Recent transcript:**

- **R1 patient (es)**: _¡Hola! Me parece bien_
  - **(en)**: _Hi! Sounds good to me_
- **R2 patient (es)**: _Pues, principalmente, porque me dan ansiedad diferentes situaciones. Por ejemplo, si tengo que viajar, sobretodo en tren, o si tengo que hablar en público o en un grupo grande. Incluso por whatsapp si es un grupo prefiero no hablar._
  - **(en)**: _Well, mainly because I get anxious in different situations. For example, if I have to travel, especially by train, or if I have to speak in public or in a large group. Even on whatsapp if it's a group I prefer not to talk._

**Candidate options:**

- **Option 1** — **GOLD** / **SUBMITTED** (es): _Entiendo. Es bastante común sentir ansiedad en situaciones como las que me describes. Viajar, hablar en público, interactuar en grupos... son contextos que pueden generar mucha activación. Es normal que te sientas así, especialmente si has tenido experiencias previas que hayan reforzado esa ansiedad. ¿Podrías contarme un poco más sobre qué es lo que sientes o piensas justo antes, durante y después de esas situaciones? No te preocupes por ser perfecta o coherente, simplemente cuéntamelo como te s_
  - **(en)**: _I understand. It is quite common to feel anxiety in situations like the ones you describe. Traveling, public speaking, interacting in groups... these are contexts that can generate a lot of activation. It's normal to feel that way, especially if you have had previous experiences that have reinforced that anxiety. Could you tell me a bit more about what you feel or think just before, during and after those situations? Don't worry about being perfect or consistent, just tell it to me as it comes o_
- **Option 2** (es): _Es normal que te sientas así, y es importante reconocer que la ansiedad puede aparecer en momentos que parecen simples o rutinarios. ¿Te has preguntado qué sucede en tu mente cuando te enfrentas a esas situaciones? ¿Qué tipo de pensamientos o sensaciones te acompañan en ese momento?_
  - **(en)**: _It is normal to feel this way, and it is important to recognize that anxiety can appear in moments that seem simple or routine. Have you asked yourself what happens in your mind when you face these situations? What kind of thoughts or sensations accompany you at that moment?_
- **Option 3** — **S2** (es): _Entiendo. Es bastante común sentir ansiedad en situaciones como las que me describes. Viajar, hablar en público, interactuar en grupos... son contextos que pueden generar mucha activación. Es normal que te sientas así, especialmente si has tenido experiencias previas que hayan reforzado esa ansiedad. ¿Podrías contarme un poco más sobre qué es lo que sientes o piensas justo antes, durante y después de esas situaciones? No te preocupes por ser perfecto o coherente, simplemente cuéntamelo como te s_
  - **(en)**: _I understand. It is quite common to feel anxiety in situations like the ones you describe. Traveling, public speaking, interacting in groups... these are contexts that can generate a lot of activation. It's normal to feel that way, especially if you have had previous experiences that have reinforced that anxiety. Could you tell me a bit more about what you feel or think just before, during and after those situations? Don't worry about being perfect or consistent, just tell it to me as it comes o_

**S2 reasoning:** _Option 3 is identical to Option 1 but corrects the gender agreement to 'perfecto', which is more appropriate for the general patient context provided._

**Notes:** _(fill in)_

---

### #8 — S06 round 12  (gold=2, submitted=2, S2=1)

**Recent transcript:**

- **R9 patient (es)**: _Si tuviera dinero como para no tener que trabajar y estudiar a la vez, mi vida sería más sencilla y estaría menos cansado. Supongo que eso me ayudaría a afrontar mejor las cosas Por ejemplo, hoy me acosté a las 2 am, que es cuando llegué a mi casa del trabajo. A las 8 am tengo clases y llego a casa _
  - **(en)**: _If I had money so that I didn't have to work and study at the same time, my life would be simpler and I would be less tired. I guess that would help me cope better For example, today I went to bed at 2 am, which is when I got home from work. At 8 am I have classes and I get home at 3:30 pm. At 6:00 _
- **R10 patient (es)**: _Sí, si a mí el dinero como tal me la pela. No es como que lo desee. Ni es el dinero en sí lo que necesito, sino el tiempo de ocio Supongo que es un medio Pero claro, es que la sociedad funciona así. Yo por mí, pues no trabajaría, pero entonces no podría pagar la casa ni la comida, ni nada No es como_
  - **(en)**: _Yes, I don't care about money as such. It's not like I want it. It's not money per se that I need, but leisure time. I suppose it's a means. I wouldn't work for me, but then I wouldn't be able to pay for the house or the food, or anything. It's not like I want the money, the landlord wants it, the g_
- **R11 patient (es)**: _Dormir. Jugar videojuegos. Verme algunas pelis que tengo pendientes... eso básicamente Si tuviera aún más tiempo, leer._
  - **(en)**: _Sleeping. Play video games. Watch some movies I have pending... that basically If I had even more time, read._
- **R12 patient (es)**: _Yo creo que es la falta de tiempo_
  - **(en)**: _I think it is the lack of time_

**Candidate options:**

- **Option 1** — **S2** (es): _¿Y qué pasaría si pudieras encontrar 15 minutos mañana para hacer algo de lo que disfrutas? ¿Qué te parece más factible?_
  - **(en)**: _What if you could find 15 minutes tomorrow to do something you enjoy? What seems more feasible?_
- **Option 2** — **GOLD** / **SUBMITTED** (es): _¿Y qué te dice tu mente cuando piensas en tomarte un rato para jugar, ver una película o leer? ¿Qué pensamientos o sensaciones surgen?_
  - **(en)**: _And what does your mind tell you when you think about taking some time to play a game, watch a movie or read? What thoughts or sensations arise?_
- **Option 3** (es): _¿Y qué pasaría si, aunque solo fuera por un rato, te permitieras hacer algo que te guste, sin preocuparte por el tiempo? ¿Te atreverías a intentarlo?_
  - **(en)**: _And what if, even if it was just for a while, you allowed yourself to do something you love, without worrying about the time? Would you dare to try it?_

**S2 reasoning:** _Given the patient's extreme exhaustion and crushing schedule, a small, concrete, and manageable goal is more realistic and less overwhelming than open-ended questions about their mind or 'not worrying about time'._

**Notes:** _(fill in)_

---

## Section C — Both wrong, disagree

_55 cases total; sampling 8 stratified by gold class._

### #1 — S01 round 23  (gold=1, submitted=2, S2=3)

**Recent transcript:**

- **R20 patient (es)**: _Pues hacer más cosas estilo, no estar constantemente preocupado por si alguien pudiera usar la parte más sensible de mi, hacer algún otro plan y tener pareja, que en el fondo me da algo de miedo, sobre todo porque en algún futuro lejano me gustaría tener hijos, pero hay veces que no vea que puede ha_
  - **(en)**: _Well, to do more stylish things, not to be constantly worried that someone might use the most sensitive part of me, to make some other plans and to have a partner, which I'm a little afraid of, especially because in the distant future I would like to have children, but sometimes I don't see that the_
- **R21 patient (es)**: _Pues quizás ese factor de desconfianza o evitar ser vulnerable Pero entonces debería ser vulnerable? Debería de dar una confianza que me puede romper en mil pedazos? Eso es muy peligroso, no sabes nunca quién te va a hacer daño_
  - **(en)**: _So maybe that factor of mistrust or avoiding being vulnerable But then should I be vulnerable? Should I give a confidence that can break me into a thousand pieces? That's very dangerous, you never know who is going to hurt you._
- **R22 patient (es)**: _Pues porque dicen que en psicología no se juzga, aunque por dentro no sé que estarás pensando. Y si sigo así, no duraré mucho, demasiados años llevo ya_
  - **(en)**: _Well, because they say that in psychology you don't judge, although inside I don't know what you're thinking. And if I go on like this, I won't last long, I've been there for too many years already._
- **R23 patient (es)**: _Pues quizás me iría mejor, pero también podria sufrir mucho más Y no me gustaría para nada_
  - **(en)**: _Well, maybe I would do better, but I could also suffer much more and I would not like it at all._

**Candidate options:**

- **Option 1** — **GOLD** (es): _Entiendo. ¿Qué crees que podría ser el costo de no hacer nada y seguir viviendo de esta manera?_
  - **(en)**: _I see. What do you think might be the cost of doing nothing and continuing to live this way?_
- **Option 2** — **SUBMITTED** (es): _¿Qué crees que podría pasar si optaras por un camino intermedio, en el que no te cierras completamente, pero tampoco te expones al máximo? ¿Qué tipo de equilibrio te gustaría encontrar entre la vulnerabilidad y la protección?_
  - **(en)**: _What do you think might happen if you chose a middle path, where you don't close yourself off completely, but also don't expose yourself to the fullest extent? What kind of balance would you like to find between vulnerability and protection?_
- **Option 3** — **S2** (es): _Entiendo. Parece que estás temiendo el dolor y la posibilidad de sufrir más, pero al mismo tiempo, estás sintiendo que no puedes seguir viviendo de esta manera. ¿Qué crees que podría ser el costo de no hacer nada y seguir viviendo de esta manera?_
  - **(en)**: _I understand. It sounds like you're dreading the pain and the possibility of suffering more, but at the same time, you're feeling like you can't keep living this way. What do you think might be the cost of doing nothing and continuing to live this way?_

**S2 reasoning:** _It validates the patient's fear of pain while simultaneously highlighting the contradiction of their current suffering, effectively using the ACT principle of creative hopelessness to motivate change._

**Notes:** _(fill in)_

---

### #2 — S09 round 15  (gold=2, submitted=3, S2=1)

**Recent transcript:**

- **R12 patient (es)**: _Sí, sentía que una vez que me subiera a exponer ya no podría irme en mitad de la exposición si me encontrase muy mal, o al menos no podría hacerlo sin que todo el mundo se diese cuenta de que me estaba pasando algo_
  - **(en)**: _Yes, I felt that once I went up to exhibit I could no longer leave in the middle of the show if I felt really bad, or at least I couldn't leave without everyone noticing something was wrong with me._
- **R13 patient (es)**: _Pues no lo sé pero no me gusta que los demás sepan que me esta pasando algo, sobre todo porque siento que es muy desproporcionado con respecto a la situación. Se van a preocupar o van a pensar que soy un exagerado y que lo que me pasa es una tontería y debería estar tranquilo, sobre todo en situacio_
  - **(en)**: _Well, I don't know but I don't like others to know that something is happening to me, especially because I feel it is very disproportionate to the situation. They are going to worry or think that I am an exaggerator and that what is happening to me is silly and I should be calm, especially in situat_
- **R14 patient (es)**: _Pues intento pensar que no pasa nada, que solo es ansiedad pero que no va a darme un ataque de ansiedad, o simplemente intento respirar hondo para calmarme un poco. Pero como rara vez funciona pues en el caso del autobús me quedo cerca de la puerta para poder salir en la siguiente parada si fuera ne_
  - **(en)**: _Well, I try to think that nothing is wrong, that it is just anxiety but that it is not going to give me an anxiety attack, or I just try to take a deep breath to calm down a little. But as it rarely works, in the case of the bus I stay close to the door so I can get out at the next stop if necessary_
- **R15 patient (es)**: _Pues imagino que no están influyendo para bien, pues me encuentro igual de mal con respecto a la ansiedad, incluso diría que estoy peor y quizás por eso me encuentre al límite, es como que siento que en cualquier situación me puede dar un ataque de ansiedad y cada vez le he cogido mas miedo_
  - **(en)**: _Well, I imagine that they are not influencing me for the better, because I feel just as bad with respect to anxiety, I would even say that I am worse and maybe that's why I am on the edge, it's like I feel that in any situation I can have an anxiety attack and I have become more and more afraid of i_

**Candidate options:**

- **Option 1** — **S2** (es): _Es lógico que te sientas así. Parece que estas estrategias, aunque te ayudan a sobrellevar las situaciones a corto plazo, no están abordando la raíz del problema y, de hecho, podrían estar reforzando el miedo y la evitación. Me gustaría que exploráramos juntos cómo funciona este ciclo de miedo y evitación, y cómo podemos romperlo. ¿Te parece bien?_
  - **(en)**: _It makes sense that you feel this way. It seems that these strategies, while helping you cope in the short term, are not addressing the root of the problem and may in fact be reinforcing the fear and avoidance. I'd like to explore together how this cycle of fear and avoidance works, and how we can break it. is that okay with you?_
- **Option 2** — **GOLD** (es): _Es como si estuvieras atrapado en un ciclo de miedo y ansiedad, donde cada vez te sientes más inseguro y vulnerable. ¿Qué crees que podrías cambiar para lidiar con la ansiedad? Porque parece que las estrategias que estás utilizando incrementan el problema_
  - **(en)**: _It's as if you are trapped in a cycle of fear and anxiety, where you feel increasingly insecure and vulnerable. What do you think you could change to deal with anxiety? Because it seems like the strategies you're using are increasing the problem._
- **Option 3** — **SUBMITTED** (es): _Es como si estuvieras atrapado en un ciclo de miedo y ansiedad, donde cada vez te sientes más inseguro y vulnerable. ¿Te parece que hay algo que te gustaría cambiar en esta situación, algo que te permita sentirte más seguro y en control?_
  - **(en)**: _It is as if you are trapped in a cycle of fear and anxiety, where you feel more and more insecure and vulnerable. Do you feel like there is something you would like to change in this situation, something that would allow you to feel more confident and in control?_

**S2 reasoning:** _It provides a clear therapeutic bridge from the patient's experience to the ACT conceptualization of the avoidance cycle while maintaining a supportive and collaborative tone._

**Notes:** _(fill in)_

---

### #3 — S15 round 3  (gold=3, submitted=2, S2=1)

**Recent transcript:**

- **R1 patient (es)**: _Holaa!!_
  - **(en)**: _Hello!_
- **R2 patient (es)**: _Bien… pues decidí recibir terapia porque últimamente estoy teniendo más problemas por mi falta de atención. Es algo que siempre he sabido pero nunca me he tratado como tal en sesión_
  - **(en)**: _Well... I decided to get therapy because lately I've been having more problems with my lack of attention. It's something I've always known but I've never treated myself as such in session._
- **R3 patient (es)**: _Pues a lo mejor cuando me cuenta algo cambio de tema continuamente, o no termino de escuchar y estoy pensando en otras cosas. Eso hace que se me olviden muchas cosas y al final quedo mal con la gente de mi alrededor._
  - **(en)**: _Well, maybe when he tells me something I keep changing the subject, or I don't finish listening and I'm thinking about other things. That makes me forget a lot of things and in the end I look bad to the people around me._

**Candidate options:**

- **Option 1** — **S2** (es): _Entiendo. Esa falta de atención parece estar afectando tu vida de maneras significativas. ¿Te has preguntado qué pasa cuando te das cuenta de que estás distrayéndote? ¿Qué sucede en ese momento?_
  - **(en)**: _I get it. That lack of attention seems to be affecting your life in significant ways. Have you ever wondered what happens when you realize you're getting distracted? What happens in that moment?_
- **Option 2** — **SUBMITTED** (es): _Entiendo. Suena frustrante, ¿verdad? Es natural sentirse mal cuando las cosas no salen como uno espera, especialmente cuando afecta las relaciones con los demás. Y es valiente que reconozcas esto y quieras trabajar en ello. ¿Qué sientes cuando te das cuenta de que estás cambiando de tema o no estás escuchando? ¿Qué pasa por tu cabeza en esos momentos?_
  - **(en)**: _I understand. Sounds frustrating, doesn't it? It's natural to feel bad when things don't go the way you expect, especially when it affects your relationships with others. And it's brave that you recognize this and want to work on it. How do you feel when you realize you're changing the subject or not listening? What's going through your head in those moments?_
- **Option 3** — **GOLD** (es): _Parece que la falta de atención está afectando tus relaciones con los demás. ¿Es como si tuviera una especie de "modo automático" que te hace cambiar de tema o dejar de escuchar sin darte cuenta? ¿Te pasa en situaciones específicas o es algo que sucede en general?_
  - **(en)**: _It seems that lack of attention is affecting your relationships with others. Is it as if you have a kind of "automatic mode" that makes you change the subject or stop listening without noticing? Does it happen to you in specific situations or is it something that happens in general?_

**S2 reasoning:** _It is direct, validates the impact on the patient's life, and immediately moves toward mindfulness/awareness of the process, which is a core component of ACT._

**Notes:** _(fill in)_

---

### #4 — S15 round 7  (gold=1, submitted=3, S2=2)

**Recent transcript:**

- **R4 patient (es)**: _También me pasa con tareas de clase, que me cuesta mucho terminar de hacer lo que realmente quería porque me distraigo muchísimo y nunca termino lo que tenía que hacer_
  - **(en)**: _It also happens to me with class assignments, that I have a hard time finishing what I really wanted to do because I get so distracted and I never finish what I had to do._
- **R5 patient (es)**: _Pues suele ser de manera general, aunque cuando es una conversación bastante importante me esfuerzo bastante y entro en un foco de atención que ya no me pasa, pero para eso como digo me tengo que esforzar bastante_
  - **(en)**: _Well, it is usually in a general way, although when it is a very important conversation I make a big effort and I get into a focus of attention that does not happen to me anymore, but for that, as I say, I have to make a big effort._
- **R6 patient (es)**: _Y realmente lo que me hace cambiar de tema, es el hecho de tener mucho pensamientos en mi cabeza a la vez mientras esa conversación está sucediendo, entonces de manera espontánea me sale decir otra cosa_
  - **(en)**: _And really what makes me change the subject, is the fact that I have a lot of thoughts in my head at the same time while that conversation is going on, so I spontaneously come up with something else to say_
- **R7 patient (es)**: _A ver realmente, para que yo la trate como una conversación importante suele pasarme poco. Son a lo mejor situaciones muy formales, o que directamente la persona ya se ha quejado de que antes no le he prestado atención entonces ya me esfuerzo en hacerlo. Aunque normalmente al momento de después de s_
  - **(en)**: _Actually, for me to treat it as an important conversation usually doesn't happen much. Maybe they are very formal situations, or the person has already complained that I haven't paid attention to them before, so I make an effort to do it. Although usually the moment after it usually happens again. I_

**Candidate options:**

- **Option 1** — **GOLD** (es): _Entiendo mejor ahora. Parece que lo que te hace esforzarte por prestar atención es el estrés o la presión de la situación, más que el interés genuino en la conversación. Es como si estuvieras tratando de evitar problemas o consecuencias en lugar de realmente conectarte con la otra persona. ¿Se parece eso a lo que te pasa?_
  - **(en)**: _I understand better now. It sounds like what makes you struggle to pay attention is the stress or pressure of the situation, rather than genuine interest in the conversation. It's as if you're trying to avoid problems or consequences rather than really connecting with the other person. Does that sound like what's going on with you?_
- **Option 2** — **S2** (es): _Entiendo mejor ahora. Es como si tu mente estuviera constantemente "saturada" de pensamientos, y eso te hace perder el enfoque en la conversación o en la tarea que estás haciendo. Es como si no pudieras "silenciar" ese ruido mental y centrarte en lo que te rodea. ¿Te parece que ese ruido mental esté relacionado con la ansiedad o la preocupación por algo en particular, o es más bien una sensación de estar "apagado" o "desconectado"?_
  - **(en)**: _I understand better now. It's as if your mind is constantly "cluttered" with thoughts, and that makes you lose focus on the conversation or task at hand. It's as if you can't "mute" that mental noise and focus on what's around you. Does that mental noise seem to you to be related to anxiety or worry about something in particular, or is it more a feeling of being "off" or "disconnected"?_
- **Option 3** — **SUBMITTED** (es): _Entiendo. Es como si tu mente estuviera en un estado de "modo de alta actividad" constante, lleno de pensamientos que se suceden uno tras otro. Y cuando algo es importante, logras enfocarte, pero luego, como si fuera una película con el volumen alto, el ruido vuelve. ¿Te gustaría explorar qué es lo que hace que esos pensamientos aparezcan con tanta frecuencia? ¿Hay algo en tu entorno o en tu historia que los alimente?_
  - **(en)**: _I get it. It's as if your mind is in a state of constant "high activity mode," full of thoughts that come one after the other, and when something is important, you manage to focus, but then, like a loud movie, the noise comes back. And when something is important, you manage to focus, but then, like a movie with the volume turned up, the noise comes back. Would you like to explore what it is that makes these thoughts appear so frequently? Is there something in your environment or in your history_

**S2 reasoning:** _It validates the patient's metaphor of 'noise' and 'saturation' while gently exploring the nature of those thoughts without jumping to conclusions or over-analyzing the motivation._

**Notes:** _(fill in)_

---

### #5 — S05 round 18  (gold=2, submitted=3, S2=1)

**Recent transcript:**

- **R15 patient (es)**: _Normalmente en mis propios pensamientos, incluso entro como en un bucle de quiero atender pero no lo estoy haciendo porque no soy capaz. Me pongo a pensar en los exámenes que tenemos la semana que viene y que todos mis compañeros llevan genial, pero ellos solo viven por y para estudiar, siempre en l_
  - **(en)**: _Normally in my own thoughts, I even get into a loop of I want to attend but I am not doing it because I am not able to. I start thinking about the exams we have next week and that all my classmates are doing great, but they only live for and to study, always in the library, and I see life in a diffe_
- **R16 patient (es)**: _Pues no sé, depende del día... de la asignatura... no me ha dado por mirar el reloj y ver cuánto tiempo pasa, no sabría decirte_
  - **(en)**: _Well, I don't know, it depends on the day... on the subject... I haven't felt like looking at the clock and seeing how much time goes by, I wouldn't know how to tell you_
- **R17 patient (es)**: _Creo que cuando llevo un rato, como que de buenas a primeras digo ostras, que me he ido y no  me he enterado de nada de lo que ha dicho... y le acabo preguntando a un compañero cercano, pero me da cosa hacerlo porque no quiero hablar en clase y molestar a mi compañero o distraer al profesor porque a_
  - **(en)**: _I think that when I've been there for a while, I kind of just go, oops, I'm gone and I didn't hear anything he said... and I end up asking a close classmate, but I don't want to do it because I don't want to talk in class and bother my classmate or distract the teacher because it was my fault in the_
- **R18 patient (es)**: _Pues como ya estoy perdida del todo, digo voy a intentar coger el hilo de la clase de nuevo y como lo intente un rato y no lo consiga lo doy por perdido y me pongo a hacer otras cosas_
  - **(en)**: _Since I'm already lost, I'm going to try to pick up the class thread again and if I try it for a while and don't succeed I'll give it up and start doing other things._

**Candidate options:**

- **Option 1** — **S2** (es): _¿Y qué sientes cuando haces eso, cuando decides "dar la clase por perdida" y te pones a hacer otra cosa? ¿Hay alguna emoción o sensación que predomine?_
  - **(en)**: _And what do you feel when you do that, when you decide to "write the class off" and go on to do something else? Is there an emotion or feeling that predominates?_
- **Option 2** — **GOLD** (es): _¿Y qué sientes cuando haces eso, cuando decides "dar la clase por perdida" y te pones a hacer otra cosa?_
  - **(en)**: _And how do you feel when you do that, when you decide to "write the class off" and move on to something else?_
- **Option 3** — **SUBMITTED** (es): _Y entonces, cuando te das cuenta de que ya no estás en el hilo, ¿qué te sucede en ese momento? ¿Te sientes frustrada, culpable, o como si algo se te hubiera escapado?_
  - **(en)**: _And then, when you realize that you are no longer in the thread, what happens to you at that moment? Do you feel frustrated, guilty, or as if something has slipped away from you?_

**S2 reasoning:** _It follows the flow of the conversation by exploring the emotional consequence of the patient's avoidance behavior, which is a key step in ACT to identify the function of the behavior._

**Notes:** _(fill in)_

---

### #6 — S09 round 24  (gold=3, submitted=2, S2=1)

**Recent transcript:**

- **R21 patient (es)**: _Sí la verdad siento que me ayudaría a recuperar un poco mi vida_
  - **(en)**: _Yes, I really feel it would help me to get my life back a little bit._
- **R22 patient (es)**: _Pues me da bastante miedo porque se que me va a dar ansiedad nada más subirme y que el autobús empiece a andar_
  - **(en)**: _Well, it scares me a lot because I know I'm going to feel anxious as soon as I get on the bus and the bus starts running._
- **R23 patient (es)**: _Pues la verdad no lo sé, lo único que se me ocurre es intentar hacer respiraciones profundas para ver si consigo tranquilizarme, pero siento que la gente podría darse cuenta_
  - **(en)**: _Well, I don't really know, the only thing I can think of is to try to take deep breaths to see if I can calm myself down, but I feel that people might notice._
- **R24 patient (es)**: _¿O sea que no hiciera nada?_
  - **(en)**: _So he did nothing?_

**Candidate options:**

- **Option 1** — **S2** (es): _Sí, exactamente. No hacer nada, simplemente permitir que la ansiedad esté ahí, sin intentar luchar contra ella, sin intentar controlarla. ¿Qué te parece la idea?_
  - **(en)**: _Yes, exactly. Doing nothing, just allowing the anxiety to be there, not trying to fight it, not trying to control it. What do you think of the idea?_
- **Option 2** — **SUBMITTED** (es): _No exactamente. Significa observar la ansiedad sin intentar cambiarla, sin juzgarla. Permitiéndote sentirla sin intentar que los demás se den cuenta. ¿Te parece posible?_
  - **(en)**: _Not exactly. It means observing anxiety without trying to change it, without judging it. Allowing yourself to feel it without trying to make others notice. Does that sound possible?_
- **Option 3** — **GOLD** (es): _No exactamente. ¿Conoces la metáfora del invitado no deseado?_
  - **(en)**: _Not exactly. You know the metaphor of the uninvited guest?_

**S2 reasoning:** _The patient is questioning the concept of non-struggle, and a direct, validating confirmation of the ACT principle of acceptance (letting the anxiety be there) is the most natural and effective way to move forward._

**Notes:** _(fill in)_

---

### #7 — S09 round 12  (gold=1, submitted=3, S2=2)

**Recent transcript:**

- **R9 patient (es)**: _Pues el corazón me iba a mil por hora, y sentía que me iba a desmayar. Y como no quería que mi peluquero se diese cuenta de lo que me estaba pasando fue la peor experiencia de mi vida soportar esa sensación hasta que acabó. Tenía mil pensamientos en la cabeza pero todos estaban relacionados con que _
  - **(en)**: _My heart was racing, and I felt like I was going to faint. And since I didn't want my hairdresser to realize what was happening to me, it was the worst experience of my life to endure that feeling until it was over. I had a thousand thoughts in my head but they were all related to the fact that I wa_
- **R10 patient (es)**: _Pues antes de este momento mi vida era normal, podía hacer lo que quisiera y era libre. A partir de ese día me dedico todo el tiempo a intentar no sufrir un ataque de ansiedad y a evitar planes y situaciones. Me he vuelto una persona muy rigida y si me sacan de mi rutina siento que todo va a hacer q_
  - **(en)**: _Well, before this moment my life was normal, I could do whatever I wanted and I was free. From that day on I spend all my time trying not to have an anxiety attack and avoiding plans and situations. I have become a very rigid person and if you take me out of my routine I feel that everything is goin_
- **R11 patient (es)**: _Pues sobre todo situaciones en las que hay gente y en las que no puedo escapar con facilidad, como estar en clase, autobuses, cortarme el pelo sigue siendo un calvario, etc. Pero hay días que incluso situaciones como comer con mi familia me provocan ansiedad y siento que no estoy presente porque est_
  - **(en)**: _Well, especially situations where there are people and where I can't escape easily, like being in class, buses, getting a haircut is still an ordeal, etc. But there are days that even situations like eating with my family cause me anxiety and I feel that I am not present because I am all the time tr_
- **R12 patient (es)**: _Sí, sentía que una vez que me subiera a exponer ya no podría irme en mitad de la exposición si me encontrase muy mal, o al menos no podría hacerlo sin que todo el mundo se diese cuenta de que me estaba pasando algo_
  - **(en)**: _Yes, I felt that once I went up to exhibit I could no longer leave in the middle of the show if I felt really bad, or at least I couldn't leave without everyone noticing something was wrong with me._

**Candidate options:**

- **Option 1** — **GOLD** (es): _Entiendo. Parece que te preocupa la idea de que los demás sean conscientes de que estás experimentando ansiedad. ¿Por qué crees que pasa esto?_
  - **(en)**: _I understand. You seem to be concerned about the idea of others being aware that you are experiencing anxiety. Why do you think this happens?_
- **Option 2** — **S2** (es): _Entiendo. Parece que la idea de perder el control en un entorno público y la preocupación por cómo te percibirían los demás te genera mucha angustia. ¿Te sientes presionado a mantener una imagen de normalidad?_
  - **(en)**: _I understand. It seems like the idea of losing control in a public setting and worrying about how others would perceive you creates a lot of angst for you. Do you feel pressured to maintain an image of normalcy?_
- **Option 3** — **SUBMITTED** (es): _Entiendo que esa sensación de no poder escapar, de no poder controlar lo que estás sintiendo en ese momento, es una experiencia muy intensa. Me gustaría que reflexionáramos juntos sobre esas situaciones donde te sientes atrapado. ¿Te parece bien que lo hagamos?_
  - **(en)**: _I understand that feeling of not being able to escape, of not being able to control what you are feeling at that moment, is a very intense experience. I would like us to reflect together on those situations where you feel trapped. Is it okay if we do that?_

**S2 reasoning:** _It accurately validates the patient's fear of public judgment and the struggle to maintain a 'normal' facade, which is a central theme in their narrative of avoidance and rigidity._

**Notes:** _(fill in)_

---

### #8 — S04 round 3  (gold=2, submitted=3, S2=1)

**Recent transcript:**

- **R1 patient (es)**: _Hola,  estoy bien algo nerviosa_
  - **(en)**: _Hi, I'm fine, I'm a little nervous_
- **R2 patient (es)**: _No sé por donde empezar, solicité la ayuda un poco por lo que comenté en el cuestionario de evaluación inicial. Tengo pensamientos muy repetitivos que me hacen estar en marcha constantemente, me cuesta mucho relajarme y siempre suelo estar buscando tareas pendientes. Cuando no tengo nada que hacer s_
  - **(en)**: _I don't know where to start, I asked for help a little bit because of what I said in the initial assessment questionnaire. I have very repetitive thoughts that make me be constantly on the go, it is very difficult for me to relax and I am always looking for pending tasks. When I have nothing to do I_
- **R3 patient (es)**: _Especialmente desde que comencé a estudiar fuera y me independicé, hará unos 4 años._
  - **(en)**: _Especially since I started studying abroad and became independent, about 4 years ago._

**Candidate options:**

- **Option 1** — **S2** (es): _Y, ¿cómo describirías tu vida antes de ese cambio, hace cuatro años? ¿Te sentías de manera diferente?_
  - **(en)**: _And how would you describe your life before that change, four years ago? Did you feel differently?_
- **Option 2** — **GOLD** (es): _¿Estas en un piso de estudiantes?_
  - **(en)**: _Are you in a student apartment?_
- **Option 3** — **SUBMITTED** (es): _La independencia puede ser un cambio importante en la vida de alguien. ¿Crees que esto tuvo algún impacto específico en tus pensamientos y sentimientos?_
  - **(en)**: _Independence can be a major change in someone's life. Do you think this had any specific impact on your thoughts and feelings?_

**S2 reasoning:** _It establishes a baseline by comparing the current state with the previous one, helping the patient identify the onset and nature of the patterns._

**Notes:** _(fill in)_

---

## Section D — Both wrong, same answer

_80 cases total; sampling 8 stratified by gold class._

### #1 — S01 round 12  (gold=1, submitted=2, S2=2)

**Recent transcript:**

- **R9 patient (es)**: _Si, todavía no había hecho la comunión Fui el único que tuvo dos y en una de esas tuve un ataque agresivo_
  - **(en)**: _Yes, I had not yet made communion I was the only one who had two and in one of those I had an aggressive attack._
- **R10 patient (es)**: _Pues si lo intento recordar, ni bien ni mal, aunque en el momento que empezó este síntoma depresivo del que me acostumbre, intenté pedir ayuda y nadie me la dio, "eran tonterías, si fuese algo serio"_
  - **(en)**: _Well, if I try to remember it, neither good nor bad, although at the moment this depressive symptom that I got used to started, I tried to ask for help and nobody gave it to me, "it was nonsense, if it was something serious"._
- **R11 patient (es)**: _Si, fue un momento difícil o eso diría, para mí no lo fue. Solo me dio un aprendizaje, no confiar en la gente, te dan la patada y solo te utilizan_
  - **(en)**: _Yes, it was a difficult time or so I would say, for me it was not. It just gave me a learning experience, don't trust people, they give you the boot and just use you._
- **R12 patient (es)**: _Quizás. Me dicen que soy muy cerrado y serio, pero los problemas son mios y yo los resuelvo, contárselo a la gente solo genera problemas o no me van a ayudar. Nadie me conoce realmente, solo partes de mi También eso me hizo difícil el hacer algo por mi mismo de algún hobbie, porque al no decir lo qu_
  - **(en)**: _Maybe. They tell me that I am very closed and serious, but the problems are mine and I solve them, telling people only generates problems or they are not going to help me. No one really knows me, only parts of me. That also made it difficult for me to do something for myself in a hobby, because when_

**Candidate options:**

- **Option 1** — **GOLD** (es): _Entiendo. Parece que has aprendido a mantener tus problemas para ti mismo y a adaptarte a las expectativas de los demás. Eso puede ser agotador a largo plazo. ¿Qué te impediría compartir tus pensamientos y sentimientos con alguien de confianza, aunque sea solo un poco?_
  - **(en)**: _I understand. It sounds like you've learned to keep your problems to yourself and adapt to the expectations of others. That can be exhausting in the long run. What would stop you from sharing your thoughts and feelings with someone you trust, even just a little?_
- **Option 2** — **SUBMITTED** / **S2** (es): _Es interesante lo que dices sobre cómo te relacionas con los demás. Parece que has desarrollado una forma de protegerte, pero también te está limitando. ¿Crees que esa forma de ser "cerrado" es algo que te ha ayudado en algún momento?_
  - **(en)**: _It's interesting what you say about how you relate to others. It sounds like you've developed a way of protecting yourself, but it's also limiting you. Do you think that way of being "closed off" is something that has helped you at some point?_
- **Option 3** (es): _Parece que has desarrollado una especie de "muro" para protegerte de ser rechazado o utilizado por los demás. Eso puede ser muy difícil para establecer relaciones verdaderas y cercanas con la gente. ¿Te parece que esta falta de confianza en la gente también te hace sentir solo o aislado?_
  - **(en)**: _You seem to have developed a kind of "wall" to protect yourself from being rejected or used by others. That can be very difficult to establish true and close relationships with people. Do you find that this lack of trust in people also makes you feel lonely or isolated?_

**S2 reasoning:** _It validates the patient's current coping mechanism as a functional protection strategy before gently exploring its current cost, which aligns with the ACT approach of observing patterns without immediate judgment._

**Notes:** _(fill in)_

---

### #2 — S16 round 18  (gold=2, submitted=1, S2=1)

**Recent transcript:**

- **R15 patient (es)**: _Intentaré fijarme a lo largo de la semana en todo eso_
  - **(en)**: _I will try to pay attention during the week to all of this._
- **R16 patient (es)**: _Sí, puedo intentarlo. ¿Entonces solo me fijo en que la he abierto y no hago nada más? ¿Solo fijarme en lo que aparece en la carpeta?_
  - **(en)**: _Yes, I can try, so I just notice that I've opened it and do nothing else? Just notice what's in the folder?_
- **R17 patient (es)**: _Vale, sí. Voy a intentarlo durante la semana._
  - **(en)**: _Okay, yes, I'm going to try during the week._
- **R18 patient (es)**: _Vale, lo tendré en cuenta_
  - **(en)**: _Okay, I'll keep that in mind_

**Candidate options:**

- **Option 1** — **SUBMITTED** / **S2** (es): _Perfecto. Y si en algún momento te sientes abrumada o te cuesta mucho, no te preocupes. Es normal. Lo importante es que te des cuenta de que estás abriendo la carpeta, y que estás ahí, presente, con esa sensación de ansiedad. Solo eso._
  - **(en)**: _Perfect. And if at some point you feel overwhelmed or have a hard time, don't worry. It's normal. The important thing is that you realize that you are opening the folder, and that you are there, present, with that feeling of anxiety. That's all._
- **Option 2** — **GOLD** (es): _Te escribo luego y quedamos para la próxima sesión. Gracias_
  - **(en)**: _I'll write you later and we'll meet for the next session. Thank you_
- **Option 3** (es): _Antes de que nos vayamos, ¿qué es algo que quieras lograr en las próximas semanas? ¿Qué es algo que te gustaría poder cambiar o mejorar en tu relación con la ansiedad, o en tus interacciones sociales?_
  - **(en)**: _Before we go, what is something you want to accomplish in the next few weeks? What is something you wish you could change or improve in your relationship with anxiety, or in your social interactions?_

**S2 reasoning:** _The patient has just agreed to a specific ACT exercise (the 'folder' metaphor for defusion/mindfulness), and the therapist needs to provide reassurance and clarify the goal of the exercise before ending the session._

**Notes:** _(fill in)_

---

### #3 — S04 round 17  (gold=3, submitted=1, S2=1)

**Recent transcript:**

- **R14 patient (es)**: _Desde niña intentar constantemente que eso cambiará y que dentro de la complejidad del problema la ayuda pudiese hacer que llevase una vida mejor o que pudiésemos llevarla juntas. Pero cuando todos los intentos fracasaron, me limite a acompañarla en su estado, a veces pasando a ser una sombra de ese_
  - **(en)**: _Since I was a child, I have constantly tried to change that and that within the complexity of the problem, help could help her lead a better life or that we could lead a better life together. But when all the attempts failed, I limited myself to accompany her in her state, sometimes becoming a shado_
- **R15 patient (es)**: _Sí, es exactamente la palabra que usaría, resentimiento. Es un sentimiento que ha subido progresivamente durante los años, de modo que a veces encuentro que esos momentos cuando salta todo por los aires son provocados por mí, como si ya no los aguantase, o tuviese algo dentro pudriéndose que me hace_
  - **(en)**: _Yes, that's exactly the word I would use, resentment. It's a feeling that has crept up progressively over the years, so that sometimes I find that those moments when it all blows up are triggered by me, like I can't take it anymore, or I have something festering inside that makes me defensive and co_
- **R16 patient (es)**: _Sí completamente culpable, porque ese resentimiento suele crear conflictos, y yo no llevo nada bien estar mal con ellos._
  - **(en)**: _Yes completely guilty, because that resentment usually creates conflicts, and I don't take well to be bad with them._
- **R17 patient (es)**: _No sé si es por la culpa o porque aunque hay una comunicación total, hay una inexistente comunicación emocional, pero no soy capaz de expresarlo con ellos. En cuanto a los pensamientos negativos, no tengo duda de que sí me hace más vulnerable a ellos._
  - **(en)**: _I don't know if it's because of guilt or because although there is total communication, there is non-existent emotional communication, but I am not able to express it with them. As for negative thoughts, I have no doubt that it does make me more vulnerable to them._

**Candidate options:**

- **Option 1** — **SUBMITTED** / **S2** (es): _Entiendo que la falta de comunicación emocional puede ser un muro difícil de cruzar, especialmente cuando hay tanto cariño y tanto dolor. Es como si, aunque puedas hablar, no puedas realmente conectarte con lo que sientes. ¿Te has sentido aislada en este proceso? ¿Crees que esa sensación de aislamiento contribuye a los pensamientos que te mencionabas antes?_
  - **(en)**: _I understand that lack of emotional communication can be a difficult wall to cross, especially when there is so much caring and so much pain. It's as if, even though you can talk, you can't really connect with what you are feeling. Have you felt isolated in this process? Do you think that feeling of isolation contributes to the thoughts you mentioned earlier?_
- **Option 2** (es): _Entiendo. Parece que existe una desconexión entre lo que sientes y cómo lo comunicas. La comunicación emocional es fundamental para cualquier relación sana, y es comprensible que te resulte difícil expresarla si no hay un espacio seguro para hacerlo. ¿Alguna vez has intentado expresar tus sentimientos a tus padres, aunque sea de forma indirecta?_
  - **(en)**: _I understand. There seems to be a disconnect between how you feel and how you communicate it. Emotional communication is fundamental to any healthy relationship, and it's understandable that you find it difficult to express it if there is no safe space to do so. Have you ever tried to express your feelings to your parents, even if it's indirectly?_
- **Option 3** — **GOLD** (es): _Dime un poco más sobre los pensamientos negativos que tienes. ¿Qué sueles pensar?_
  - **(en)**: _Tell me a little more about the negative thoughts you have. What do you usually think?_

**S2 reasoning:** _It validates the patient's complex emotional state and connects the current discussion about family dynamics back to the initial complaint of negative thoughts, maintaining the therapeutic flow._

**Notes:** _(fill in)_

---

### #4 — S04 round 2  (gold=1, submitted=3, S2=3)

**Recent transcript:**

- **R1 patient (es)**: _Hola,  estoy bien algo nerviosa_
  - **(en)**: _Hi, I'm fine, I'm a little nervous_
- **R2 patient (es)**: _No sé por donde empezar, solicité la ayuda un poco por lo que comenté en el cuestionario de evaluación inicial. Tengo pensamientos muy repetitivos que me hacen estar en marcha constantemente, me cuesta mucho relajarme y siempre suelo estar buscando tareas pendientes. Cuando no tengo nada que hacer s_
  - **(en)**: _I don't know where to start, I asked for help a little bit because of what I said in the initial assessment questionnaire. I have very repetitive thoughts that make me be constantly on the go, it is very difficult for me to relax and I am always looking for pending tasks. When I have nothing to do I_

**Candidate options:**

- **Option 1** — **GOLD** (es): _Desde cuando te pasa?_
  - **(en)**: _How long has this been happening to you?_
- **Option 2** (es): _Entiendo, parece que estás experimentando síntomas de ansiedad y estrés crónico. ¿Puedes decirme más sobre estos pensamientos negativos que te generan nervios? ¿Son pensamientos específicos sobre tu vida o más bien una sensación general de inseguridad?_
  - **(en)**: _I understand, it sounds like you are experiencing symptoms of anxiety and chronic stress, can you tell me more about these negative thoughts that are making you nervous, are they specific thoughts about your life or more of a general feeling of insecurity?_
- **Option 3** — **SUBMITTED** / **S2** (es): _Entiendo. Parece que experimentas una combinación de inquietud, necesidad de actividad y pensamientos negativos que te dificultan la relajación. Es valiente de tu parte reconocer esto y buscar ayuda. Cuéntame más sobre estos pensamientos repetitivos. ¿Podrías darme un ejemplo de alguno de ellos? Y cuando dices "ideas muy negativas", ¿qué tipo de pensamientos son esos?_
  - **(en)**: _I understand. You seem to be experiencing a combination of restlessness, need for activity, and negative thoughts that make it difficult for you to relax. It is brave of you to recognize this and seek help. Tell me more about these repetitive thoughts - could you give me an example of one of them? And when you say "very negative thoughts," what kind of thoughts are those?_

**S2 reasoning:** _It validates the patient's experience and uses a gentle, open-ended approach to gather specific examples of the thoughts, which is essential for the functional analysis in ACT._

**Notes:** _(fill in)_

---

### #5 — S06 round 8  (gold=2, submitted=1, S2=1)

**Recent transcript:**

- **R5 patient (es)**: _Pues un poco todo. Lo noto mucho en las cosas más pequeñas y cotidianas. El hecho de levantarme de la cama es un esfuerzo muy consciente que tengo que hacer. Ponerme a cocinar algo para comer... es que pudo tener hambre, pero tengo literalmente cero ganas de emprender la tarea de "cocinar". Se me ha_
  - **(en)**: _Well, a little bit of everything. I notice it a lot in the smallest, everyday things. Getting out of bed is a very conscious effort I have to make. Cooking something to eat... I may have been hungry, but I literally have zero desire to undertake the task of "cooking". It's a world away. Even the lig_
- **R6 patient (es)**: _Eso es complicado. Supongo que acometería la cotidianidad con otra perspectiva. La verdad es que se me hace difícil de imaginar porque no creo que lo haya experimentado nunca Para mí, esto es lo normal. Y no sé, porque igual todo el mundo lo siente igual, a lo mejor lo que está mal es mi percepción _
  - **(en)**: _That's complicated. I guess I would approach everyday life with a different perspective. The truth is that it's hard for me to imagine because I don't think I've ever experienced it. And I don't know, because maybe everyone feels the same way, maybe what is wrong is my perception of others. But from_
- **R7 patient (es)**: _La verdad es que no lo sé. Más allá de mi propia experiencia interna. Esto es de perogrullo, pero supongo que si pudiera disfrutar más de las cosas, pues disfrutaría más y no estaría amargado ¿Cómo cambiaría eso mi vida? Pues no lo sé, igual no lo haría_
  - **(en)**: _The truth is that I don't know. Beyond my own internal experience. This is a truism, but I suppose that if I could enjoy things more, I would enjoy more and I would not be bitter. Well, I don't know, maybe I wouldn't._
- **R8 patient (es)**: _Dinero No te digo ser rico, pero sí dejar de ser pobre_
  - **(en)**: _Money I'm not telling you to be rich, but I'm telling you to stop being poor._

**Candidate options:**

- **Option 1** — **SUBMITTED** / **S2** (es): _Entiendo. Tener más dinero parece algo que te aliviaría. ¿Qué te diría esa mayor estabilidad económica que ahora no tienes? ¿Qué posibilidades abriría?_
  - **(en)**: _I understand. Having more money seems like something that would ease your mind. What would that greater financial stability tell you that you don't have now? What possibilities would it open up?_
- **Option 2** — **GOLD** (es): _Entiendo. ¿Qué significaría para ti tener más dinero? ¿Qué cosas podrías hacer o experimentar que ahora no puedes?_
  - **(en)**: _What would it mean to you to have more money? What things could you do or experience that you can't now?_
- **Option 3** (es): _El dinero puede ser un alivio en muchos sentidos. ¿Pensando en la vida que podrías tener con un poco más de estabilidad económica, ¿qué es algo que te gustaría hacer o experimentar que ahora no puedes?_
  - **(en)**: _Money can be a relief in many ways. Thinking about the life you could have with a little more financial stability, what is something you would like to do or experience that you can't now?_

**S2 reasoning:** _It validates the patient's need for relief while gently probing the functional meaning of money, moving toward values and possibilities without sounding like a generic questionnaire._

**Notes:** _(fill in)_

---

### #6 — S12 round 6  (gold=3, submitted=2, S2=2)

**Recent transcript:**

- **R3 patient (es)**: _No he conseguido dormir del todo bien esta noche, encima esta mañana tenía unas tareas pendientes y me han surgido unos imprevistos que me han hecho atrasarlas... Me frustra el no poder haber terminado lo que tenía pensado para hoy_
  - **(en)**: _I have not been able to sleep well tonight, on top of that I had some pending tasks this morning and some unforeseen events have made me delay them.... I'm frustrated that I couldn't finish what I had planned for today._
- **R4 patient (es)**: _me genera incomodidad, no puedo estar quieta y necesito moverme de un lado para el otro todo el tiempo. Aparte siento que se me van a acumular las cosas que tengo que hacer y eso me pone peor_
  - **(en)**: _It makes me uncomfortable, I can't stand still and I need to move from one side to the other all the time. Besides, I feel that the things I have to do are going to pile up and that makes me feel worse._
- **R5 patient (es)**: _lo que más me preocupa es que no me de tiempo a terminarlas dentro del plazo y que ello pueda perjudicarme_
  - **(en)**: _what worries me the most is that I will not have time to finish them within the deadline and that this could be detrimental to me._
- **R6 patient (es)**: _No me gustaría quedar mal frente a los profesores y tampoco que mis notas bajen. Además, que muchas veces para intentar adelantar tareas ni siquiera salgo de casa en días. El único descanso que tengo es cuando me voy a dormir_
  - **(en)**: _I wouldn't want to look bad in front of the teachers and I wouldn't want my grades to go down either. Also, I often don't even leave the house for days to try to get ahead on homework. The only break I get is when I go to sleep._

**Candidate options:**

- **Option 1** (es): _Entiendo que te preocupa tu imagen ante los profesores y que no quieras que tus notas bajen. Pero también me parece que hay un patrón aquí que podría estar afectando tu bienestar. Salir de casa cada día, aunque sea para ir a la universidad, puede ser una forma de separarte de la rutina y de la ansiedad que te rodea en casa. ¿Podrías reflexionar sobre por qué te cuesta tanto salir de casa y enfrentar lo que te está pasando?_
  - **(en)**: _I understand that you are concerned about your image with teachers and that you don't want your grades to drop. But it also seems to me that there is a pattern here that could be affecting your well-being. Getting out of the house each day, even if it's just to go to college, can be a way to separate yourself from the routine and anxiety that surrounds you at home. Could you reflect on why it's so hard for you to get out of the house and face what's happening to you?_
- **Option 2** — **SUBMITTED** / **S2** (es): _Parece que hay varias capas en esa preocupación. El temor a la evaluación de los profesores, el impacto en tus notas y esa sensación de aislamiento... todo eso está conectado. Y me dices que para intentar adelantar tareas, sacrificas tu tiempo libre y tu descanso. ¿Cómo te hace sentir esa falta de equilibrio entre el trabajo y el descanso?_
  - **(en)**: _There seem to be several layers to that concern. The fear of teacher evaluation, the impact on your grades, and that feeling of isolation...all of that is connected. And you tell me that in order to try to get ahead on assignments, you sacrifice your free time and rest. How does that lack of balance between work and rest make you feel?_
- **Option 3** — **GOLD** (es): _Es importante que te sientas en contacto con tus preocupaciones, y es válido que te preocupe no quedar mal o afectar tus notas. La idea de que no salgas de casa para adelantar tareas también refleja un esfuerzo por controlar lo que te ocurre. ¿Qué te parece si exploramos juntos formas de hacer más manejable esta carga? ¿Te gustaría que te ayudara a encontrar estrategias para organizar mejor tus tareas o para manejar esos momentos de incomodidad y necesidad de moverte?_
  - **(en)**: _It's important to feel in touch with your concerns, and it's valid to be concerned about not looking bad or affecting your grades. The idea that you don't leave the house to get ahead on tasks also reflects an effort to control what happens to you. How about exploring together ways to make this burden more manageable? Would you like me to help you find strategies to better organize your tasks or to manage those moments of discomfort and need to move?_

**S2 reasoning:** _It validates the patient's multiple concerns and uses an open-ended question to explore the emotional impact of her lack of balance, maintaining a natural flow without jumping too quickly into problem-solving._

**Notes:** _(fill in)_

---

### #7 — S04 round 24  (gold=1, submitted=3, S2=3)

**Recent transcript:**

- **R21 patient (es)**: _Vale perfecto, muchas gracias._
  - **(en)**: _Ok perfect, thank you very much._
- **R22 patient (es)**: _Buenos días, estoy bien_
  - **(en)**: _Good morning, I am well_
- **R23 patient (es)**: _Me sentí un poco revuelta al hablar de temas que no suelo tocar mucho._
  - **(en)**: _I felt a little revolted talking about topics that I don't usually touch on much._
- **R24 patient (es)**: _Me levanto por la mañana, desayuno y suelo dejar el cuarto ordenado y ventilado antes de irme. Luego voy a clase, cuando llego hago tareas de la casa si hay algo pendiente y si no suelo ponerme con trabajos o estudios, si no hay trabajos o estudios trato de leer o ver alguna serie para mantenerme oc_
  - **(en)**: _I get up in the morning, have breakfast and I usually leave the room tidy and ventilated before I leave. Then I go to class, when I arrive I do homework if there is something pending and if not I usually start with work or studies, if there is no work or studies I try to read or watch a series to ke_

**Candidate options:**

- **Option 1** — **GOLD** (es): _Gracias por la información. A veces, nos pasa eso, que no podemos concentrarnos en las tareas y más si conllevan un esfuerzo de atención y concentración. ¿Sueles dedicar tiempo a algo que te guste, hobby?_
  - **(en)**: _Thanks for the information. Sometimes, that happens to us, that we can not concentrate on the tasks and more if they involve an effort of attention and concentration. Do you usually spend time on something you like, hobby?_
- **Option 2** (es): _Es interesante lo que me cuentas. Me gusta que seas consciente de tus rutinas y de cómo te organizas. Y también me gusta que seas honesta al mencionar que a veces te cuesta concentrarte. ¿Qué te parece si vamos a ver si podemos explorar un poco más esos momentos en los que te cuesta concentrarte? ¿Te gustaría hablar de eso?_
  - **(en)**: _It is interesting what you tell me. I like that you are aware of your routines and how you organize yourself. And I also like that you are honest in mentioning that sometimes you find it hard to concentrate. What do you think if we can see if we can explore a little bit more those moments when you find it hard to concentrate? Would you like to talk about that?_
- **Option 3** — **SUBMITTED** / **S2** (es): _Gracias por compartir eso conmigo. Parece una rutina bastante estructurada y organizada, lo cual es positivo. Es bueno que le dediques tiempo al orden y al cuidado de tu espacio personal, eso puede ser muy reconfortante. Me has mencionado que a veces te encuentras con dificultad para concentrarte, especialmente cuando recurres al móvil. ¿Podrías contarme un poco más sobre cómo te sientes en esos momentos? ¿Qué pensamientos o sensaciones te llevan a tomar el móvil en lugar de concentrarte en lo q_
  - **(en)**: _Thanks for sharing that with me. It sounds like a pretty structured and organized routine, which is a good thing. It's good that you make time for tidying up and taking care of your personal space, that can be very comforting. You mentioned to me that sometimes you find it difficult to concentrate, especially when you turn to your cell phone. Could you tell me a bit more about how you feel at those times? What thoughts or feelings lead you to reach for your cell phone instead of concentrating on_

**S2 reasoning:** _It validates the patient's structure while gently guiding the conversation toward the functional analysis of the avoidance behavior (using the phone), which is a key step in ACT._

**Notes:** _(fill in)_

---

### #8 — S04 round 21  (gold=2, submitted=1, S2=1)

**Recent transcript:**

- **R18 patient (es)**: _Pues suelo pensar que son mayores, que pronto morirán y que yo no habré disfrutado lo suficiente con ellos, porque sus enfermedades o hábitos lo hacen imposible, y ahora, por la distancia._
  - **(en)**: _Well, I usually think that they are old, that they will die soon and that I will not have enjoyed enough with them, because their illnesses or habits make it impossible, and now, because of the distance._
- **R19 patient (es)**: _Esos y los de mi propia muerte, siempre muerte de los demás o propia, ambos implican no verles más._
  - **(en)**: _Those and those of my own death, always death of others or my own, both imply not seeing them anymore._
- **R20 patient (es)**: _No, en absoluto, es completo terror a que ocurra, así que jamás la buscaría por mi cuenta._
  - **(en)**: _No, not at all, it's complete terror of it happening, so I would never look for it on my own._
- **R21 patient (es)**: _Vale perfecto, muchas gracias._
  - **(en)**: _Ok perfect, thank you very much._

**Candidate options:**

- **Option 1** — **SUBMITTED** / **S2** (es): _No te preocupes, estaré aquí. Si necesitas algo, no dudes en escribirme. Hasta mañana._
  - **(en)**: _Don't worry, I'll be here. If you need anything, don't hesitate to write me. See you tomorrow._
- **Option 2** — **GOLD** (es): _A ti_
  - **(en)**: _To you_
- **Option 3** (es): _De nada, me alegra poder ayudarte. Hasta mañana._
  - **(en)**: _You're welcome, I'm happy to help you. See you tomorrow._

**S2 reasoning:** _The patient has shared deep emotional pain, fear of death, and family trauma; Option 1 provides the necessary warmth and therapeutic containment for a closing statement._

**Notes:** _(fill in)_

---
