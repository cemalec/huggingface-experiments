# Checkpoint Comparison — Eval Set (20 examples)

Columns: **Source / Instruction** then one output per checkpoint.

---

## 1. Phrase as a personal letter from the 1920s reporting an event from today.

*Axes: register, tone · Tier: hard*


**Source**

> The 26-year-old was sent to warm up during the half-time break on his own before replacing Ander Herrera.


**Base model**

> My dearest friend,

I hope this letter finds you well. I am writing to share with you an exciting development in my life that has been keeping me busy lately.

Last night, at the match between our beloved team and their rivals, something extraordinary happened. During halftime, they decided to send one of us out for some light exercise. It was quite a surprise, but we all had great fun warming up together.

However, things took an unexpected turn when it came time to replace Ander Herrera. This young man, who is only 26 years old, stepped forward without hesitation. He didn't need any encouragement or instructions; he simply knew what needed to be done. His quick thinking and determination were truly inspiring.

This experience taught me a lot about teamwork and leadership. It's amazing how much can be accomplished when everyone puts their best foot forward. I'm grateful to have had such an opportunity to learn from him and to see firsthand how important it is to stay focused and committed to your goals.

Please let me know if you would like to hear more stories from my adventures. Until then, keep smiling and staying strong!

Yours sincerely,
[Your Name]

**Epoch 1**

> My dear friend, I write to you with some news of our team's proceedings this afternoon — and it concerns the young man who has been warming himself at the side of the field since the first whistle had rung. He did so alone, without any company save for himself, until the interval came round, when he took his place in the starting line-up in place of Ander Herrera. I trust you will find this report of interest. Yours faithfully, [Your Name].


**Epoch 3**

> My dearest friend, I write to you with some news of our young man's progress this afternoon — he has been dispatched alone into the dressing room for a short walk at the interval, and then, when it is time to replace Ander Herrera in the starting line-up, he has arrived ready and prepared. I trust you will find this report of interest. Yours faithfully, a member of the squad.

---

## 2. Recast as a weather-report line.

*Axes: genre · Tier: easy*


**Source**

> She started diving from a lower height, but she soon wanted to go higher.


**Base model**

> She began her dive from a shallower depth, but quickly desired to ascend further.

**Epoch 1**

> Conditions have shifted; expect a gradual increase in altitude as the diver's approach becomes more ambitious.


**Epoch 3**

> Conditions have shifted: initial low-pressure systems moved in, but now high-pressure fronts are moving in faster than expected.

---

## 3. Phrase as an audio-tour script line a museum visitor would hear in headphones.

*Axes: genre · Tier: easy*


**Source**

> Before I start I was wondering if it was ok if I put a small crescendo in here and then decrescendo back to pianissimo again over here?


**Base model**

> Before we begin, just for your comfort, I'd like to add a gentle crescendo here, and then bring it down to a soft pianissimo level there.

**Epoch 1**

> Notice how this section begins with a subtle crescendo—a momentary rise in volume—and concludes with a gentle decrescendo back down to its original softness—showing you how dynamics can guide the listener's experience through the piece.


**Epoch 3**

> Before we begin, please note that this section features a subtle dynamic shift—a crescendo followed by a gradual diminuendo—allowing you to experience the piece's full range of expression.

---

## 4. Phrase as a horoscope for a Capricorn.

*Axes: genre · Tier: easy*


**Source**

> I also used it to keep her on her side because she would sometimes spit up in her sleep and I didn't want her to choke.


**Base model**

> Capricorns are known for their determination and practicality. In your case, you have shown great foresight by using the object to prevent choking incidents during sleep. This is a sign of good judgment and responsibility, which will help you achieve success in all aspects of life. Keep this positive attitude and continue to make wise decisions.

**Epoch 1**

> Capricorn: Your resolve is tested when you must protect your charge from harm. A gentle reminder of gravity will guide you through moments of vulnerability.


**Epoch 3**

> Capricorn (Dec 22–Jan 19): Your practical instincts will guide you through challenging moments. A child's safety is your priority—keep them stable when they're vulnerable.

---

## 5. Recast as a wedding toast in the voice of a nature-documentary narrator.

*Axes: genre, voice · Tier: hard*


**Source**

> Walter Dean Myers is an acclaimed author who writes for juveniles.


**Base model**

> Walter Dean Myers, renowned author and storyteller, brings tales of adventure and heartache to young readers around the world. His words have touched countless lives, weaving magic into the fabric of childhood literature. May his legacy continue to inspire future generations with his profound insights and unforgettable narratives.

**Epoch 1**

> And here we observe something remarkable — a writer whose name has become synonymous with the young, whose work speaks directly and powerfully to those who have yet to find their own voices. Walter Dean Myers, it seems, was born to this purpose from the moment he first picked up a pen. May his legacy continue to inspire generations to come. Cheers!


**Epoch 3**

> And here we observe what few people realize: Walter Dean Myers — this remarkable creature of letters — has spent his life writing for children, and it is precisely that dedication that makes him so extraordinary. Ladies and gentlemen, please raise your glasses.

---

## 6. Phrase as a police-blotter entry in a deadpan tone.

*Axes: genre, tone · Tier: hard*


**Source**

> The thermodynamic structure that allows the flux tube to be in mechanical equilibrium is also calculated.


**Base model**

> **Police-Blotter Entry**

Date: [Insert Date]

Location: [Insert Location]

Incident Description:
A team of physicists has discovered an anomaly within their research facility. Specifically, they have identified a thermodynamic structure capable of maintaining flux tubes in mechanical equilibrium. This finding could potentially revolutionize our understanding of particle physics and quantum mechanics.

Investigation Status:
The investigation is ongoing. Further details will be provided once all relevant data has been analyzed.

[End of Entry]

**Epoch 1**

> Thermodynamic structure identified and confirmed; flux tube found in mechanical equilibrium. No further action required at this time.


**Epoch 3**

> At approximately 1400 hours on [date], thermodynamic structure was determined and confirmed to allow flux tube to remain in mechanical equilibrium. No further action required at this time.

---

## 7. Recast in a defensive tone, as if anticipating an accusation.

*Axes: tone · Tier: easy*


**Source**

> The deleted scenes are fantastic and one of them made me fall off the couch and cry, I was laughing so hard (Jay and Silent Bob's Fat Albert scene).


**Base model**

> The hidden gems within this film are truly remarkable, with one particular segment that elicited such strong emotions from me - it caused me to lose my balance on the couch and break into tears out of sheer laughter (the Jay and Silent Bob's "Fat Albert" sequence).

**Epoch 1**

> Look, these deleted scenes are genuinely great—there's even one that actually moved me to tears! And yes, it was Jay and Silent Bob's Fat Albert scene, but honestly, I laughed until my sides hurt.


**Epoch 3**

> Look, these deleted scenes are genuinely excellent—yes, even the ones that made you laugh until you cried—and yes, the Jay and Silent Bob scene with Fat Albert is genuinely funny, which is why it worked for you.

---

## 8. Phrase with condescending politeness: outwardly courteous, underneath dismissive.

*Axes: tone · Tier: easy*


**Source**

> An Italian Dario Argento knock-off but without Argentos' style and talent.The flimsy plot is just an excuse for amputations, dis-embowelments,throat cutting, shots to the head, knifings and a 8 year old cannibal.


**Base model**

> Your review of this film is quite polite on the surface, yet it's clear beneath your words that you have little respect or appreciation for the work. The plot seems rather thin, designed merely as an excuse for graphic violence such as amputations, dismemberment, throat slitting, gunshot wounds, knife stabbings, and even cannibalism involving a young child.

**Epoch 1**

> How charming—another Argento ripoff, naturally, though one must admit it lacks his distinctive flair. The story itself is merely pretext for these rather... unfortunate scenes.


**Epoch 3**

> How charming that you've found this rather pedestrian affair—a mere imitation of Argento's work—though one must admit it lacks his considerable craftsmanship. The plot itself serves merely as pretext for these unfortunate scenes involving amputation, disembowelment, throat-cutting, gunshot wounds, knife injuries, and the aforementioned young cannibal.

---

## 9. Phrase as a single haiku-like compression of seventeen syllables.

*Axes: genre, length · Tier: hard*


**Source**

> We theoretically analyzed the microwave-induced modification of the Hall magnetoresistivity in high mobility two-dimensional electron systems.


**Base model**

> Theoretical analysis,
Microwaves alter electron behavior,
Hall effect changes.

**Epoch 1**

> Theoretical analysis — / Microwave modifies Hall's resistance — / Two-dimensional electrons flow.


**Epoch 3**

> Microwaves twist the electrons — / their resistivity changes, flows shift — / two dimensions now flow.

---

## 10. Phrase as a cliffhanger that creates obligation in the reader.

*Axes: structural, tone · Tier: hard*


**Source**

> Today, the area is policed by the successor to Berkshire Constabulary, Thames Valley Police.


**Base model**

> Today marks the beginning of a new era for policing in this region, with the arrival of the successor to Berkshire Constabulary and the establishment of Thames Valley Police. What will be their approach to maintaining public safety? Stay tuned!

**Epoch 1**

> Today, the area is policed — and no one knows who will be there tomorrow. The successor to Berkshire Constabulary has taken over, but what comes next?


**Epoch 3**

> Today, the area was handed over — and now it belongs to someone new. The successor to Berkshire Constabulary has become Thames Valley Police. What happens next will change everything.

---

## 11. Translate into the dense, definition-laden register of a mathematics paper.

*Axes: register · Tier: easy*


**Source**

> Other bath tub toys have come and gone but these have kept her attention and brought lots of enjoyment- you can't go wrong with these little guys.


**Base model**

> Other bathtub toys have made their appearance and subsequently disappeared from view; however, this particular set has captivated her interest and provided ample entertainment - it is evident that one would not be disappointed in acquiring such diminutive delights.

**Epoch 1**

> Let T denote the set of all bath-tub toy configurations; let S be the subset comprising those configurations that maintain sustained visual interest and generate substantial pleasure output. The present configuration C satisfies both conditions: it remains in place throughout observation intervals, and its cumulative enjoyment metric exceeds any alternative configuration within T. Thus, one may confidently assert that C is optimal for this particular application domain.


**Epoch 3**

> Let T denote the set of bath-tub toy alternatives; we observe that ∀t ∈ T, t → ∅ → ∅ (i.e., no recurrence). However, this particular subset S ⊆ T maintains persistent engagement and produces substantial utility output. Consequently, one may assert with confidence that the collection {S} constitutes an optimal choice for young children's playtime.

---

## 12. Recast as a 404-page line in the voice of a noir detective.

*Axes: genre, voice · Tier: hard*


**Source**

> An attorney for Abraham Pishevar's family says the 18-year-old posted to social media messages entitled'Rush' in which he documented the doomed plane's ascent August 25--Pishevar's first day of college classes.


**Base model**

> "Rush," the cryptic hashtag scribbled across his laptop screen on that fateful morning of August 25th, marked the beginning of an odyssey into the unknown. The words "rush" and "college" danced like shadows through the digital ether, leaving behind only whispers of a journey fraught with peril and promise. Now, as the sun set over the campus, casting long shadows across the empty halls, the lawyer stood before the judge, her heart pounding against her ribcage like a drumroll in a silent symphony. The case was about to be decided not just on the evidence at hand but also on the weight of a name whispered under the cover of darkness—Abraham Pishevar.

**Epoch 1**

> The page you're looking for doesn't exist anymore — not on any platform that matters. The lawyer said it was called 'Rush,' and he said it went up on August 25th, right after Pishevar walked into his first class at college. But now it's gone, just like everything else.


**Epoch 3**

> The page you're looking for isn't here anymore — not on Facebook, not on Twitter, not anywhere. But I know what happened that day, and it wasn't pretty. The kid was just trying to document his own death. He called it 'Rush.' That's how they do things around here.

---

## 13. Render as a teacher's chalkboard summary at the end of a lecture.

*Axes: voice, length · Tier: hard*


**Source**

> The Cristo de la Expiración is a 17th-century Roman Catholic devotional wooden image enshrined in the titular Convento de Santo Domingo, in Cartagena de Indias, Colombia, depicting crucified Jesus.


**Base model**

> **Teacher's Chalkboard Summary**

The **Cristo de la Expiración**, a 17th-century Roman Catholic devotional wooden image, is enshrined within the **Convento de Santo Domingo** in Cartagena de Indias, Colombia. This iconic piece depicts Jesus Christ crucified, embodying profound religious devotion and historical significance.

**Epoch 1**

> Key points:
• Devotional image (Cristo de la Expiración)
• 17th century
• Roman Catholic tradition
• Wooden representation
• Enshrined in Convento de Santo Domingo
• Located in Cartagena de Indias, Colombia
• Depicts crucified Jesus


**Epoch 3**

> Key points:
• Devotional image — 17th century.
• Roman Catholic origin.
• Wooden material.
• Enshrined in Convento de Santo Domingo.
• Located in Cartagena de Indias, Colombia.
• Depicts crucified Jesus.

---

## 14. Expand by replacing each abstract noun with the concrete situation it describes.

*Axes: length · Tier: easy*


**Source**

> Senior FA figures are braced for a furious backlash if, as expected, they fail to charge Malky Mackay and Iain Moody over their exchange of sexist, racist and homophobic texts and emails.


**Base model**

> The senior football association officials are prepared for a heated criticism if, as anticipated, they do not demand severe punishments from Malky Mackay and Iain Moody regarding their offensive messages and emails containing sexist, racial, and homophobic content.

**Epoch 1**

> Senior FA officials are prepared for an intense public reaction if, as anticipated, they do not discipline two men—Malky Mackay and Iain Moody—for exchanging offensive messages that included sexual harassment, racial slurs, and homophobic language in written correspondence.


**Epoch 3**

> Senior FA officials anticipate that fans will erupt in fury when they learn that the governing body has failed to discipline two players—Malky Mackay and Iain Moody—for exchanging abusive messages online and off: words like 'nigger,' 'faggot,' and 'homophobic' sent through email and text.

---

## 15. Recast in a vulnerable tone that admits the speaker doesn't quite have this figured out.

*Axes: tone · Tier: easy*


**Source**

> King's genius is undisputable, but this book is testimony that his genius is not limited.


**Base model**

> Ah, I must admit, I'm still trying to fully grasp just how profound and far-reaching the king's genius truly is. It's fascinating to see how he can apply such incredible insight across various fields, yet it feels like there's more to uncover. Perhaps one day, I'll be able to unravel all the mysteries behind his brilliance!

**Epoch 1**

> I think I'm starting to understand—King's genius is undeniable, and yet there's something here about how he keeps pushing beyond what we expect from him, which makes me wonder if maybe it isn't just one kind of brilliance after another.


**Epoch 3**

> I think I'm starting to understand—King's genius is undeniable, and yet... maybe he isn't just one thing? This book suggests there's more to him than meets the eye.

---

## 16. Render as a personal ad placed in a print newspaper.

*Axes: genre, voice · Tier: hard*


**Source**

> The simulations include self-consistent effects of finite Alfven wave propagation and dissipation.


**Base model**

> Looking for an experienced engineer with expertise in plasma physics? We're seeking someone who can work on advanced simulation models that incorporate realistic representations of finite Alfven waves. If you have a passion for cutting-edge research and enjoy working with complex mathematical equations, this could be the perfect opportunity for you! Please contact us at [insert email or phone number] for more information.

**Epoch 1**

> Self-consistent simulation sought — one that includes both finite Alfven wave propagation and dissipation. No prior experience required. Contact for more information.


**Epoch 3**

> Self-consistent, finite-Alfven-wave-propagating, dissipative system seeking partner willing to account for all relevant effects — no hand-waving allowed. Simulations welcome.

---

## 17. Render as an oral-history fragment delivered in a tired, weary tone.

*Axes: voice, tone · Tier: hard*


**Source**

> Nam Ji-hyun as Eun Bong-hee Choi Myung-bin as young Bong-hee A prosecutor trainee at Ji-wook's office, later lawyer who was a Taekwondo athlete in her youth.


**Base model**

> Ahem... I'm not sure where this is going, but let me try my best. So there was this prosecutor trainee named Nam Ji-hyun, and she became a lawyer later on. She used to be a Taekwondo athlete when she was younger. And then she worked at Ji-wook's office as a prosecutor.

**Epoch 1**

> She was Eun Bong-hee Choi — that's what they called her. She worked for him, I think, at his office. He was a prosecutor, and then he became a lawyer. And she'd been a Taekwondo athlete when she was younger. That's how it went down.


**Epoch 3**

> She was Eun Bong-hee — that's Nam Ji-hyun, right? — and she came from Choi Myung-bin, who played Bong-hee when he was younger. He was a prosecutor trainee there too, you know, before he became a lawyer. And then... yeah, she went on to be a taekwondo athlete first, which is what I always wondered about. But that's how it goes.

---

## 18. Render as a single concise tweet aimed at a hostile cross-examiner.

*Axes: genre, audience · Tier: hard*


**Source**

> The Pullenvale Ward is a Brisbane City Council ward covering Pullenvale, Anstead, Bellbowrie, Brookfield, Karana Downs, Kenmore, Kenmore Hills, Kholo, Lake Manchester, Moggill, Mt Crosby, Pinjarra Hills, Upper Brookfield and parts of Chapel Hill and Chuwar.


**Base model**

> The Pullenvale Ward covers 14 suburbs in Brisbane City Council - not just any ward, but one with significant influence! #BrisbaneCouncil #PullenvaleWard

**Epoch 1**

> The Pullenvale Ward covers 12 wards in Brisbane City — including Pullenvale, Anstead, Bellbowrie, Brookfield, Karana Downs, Kenmore, Kenmore Hills, Kholo, Lake Manchester, Moggill, Mt Crosby, Pinjarra Hills, Upper Brookfield, and parts of Chapel Hill and Chuwar. That's not an argument; that's just facts.


**Epoch 3**

> The Pullenvale Ward covers 12 suburbs in Brisbane City Council — yes, including parts of Chapel Hill and Chuwar. That's all there is to it.

---

## 19. Render in a defiant tone that refuses to concede the framing.

*Axes: tone · Tier: easy*


**Source**

> Who can say if we'll be here tomorrow or not?


**Base model**

> Who dares tell us our fate is sealed for today? We shall forge ahead with defiance and resilience!

**Epoch 1**

> We cannot predict what tomorrow will bring—nor should anyone else.


**Epoch 3**

> We cannot predict what tomorrow will bring—nor should anyone else claim certainty over our future existence.

---

## 20. Render as an internal Slack message marked "for visibility only".

*Axes: register, genre · Tier: hard*


**Source**

> The translational energy of desorbed molecules was measured by time-of-flight (ToF) mass spectrometry.


**Base model**

> For visibility only: The translational energy of desorbed molecules was measured using time-of-flight (ToF) mass spectrometry.

**Epoch 1**

> FYI / for visibility only — just confirming that the translational energy of desorbed molecules was measured using TOF mass spectrometry. No action needed from this thread.


**Epoch 3**

> fyi / for visibility only — just confirming that the translational energy of the desorbed molecules has been measured using TOF mass spectrometry. no action needed from this thread.
