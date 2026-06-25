# Baseline Evaluation — Qwen2.5-1.5B-Instruct (no LoRA)

Compare `Base model output` against the same prompt after LoRA training.

---

## 1. Phrase as a personal letter from the 1920s reporting an event from today.

*Domain: news · Axes: register, tone · Tier: hard*

**Source**

> The 26-year-old was sent to warm up during the half-time break on his own before replacing Ander Herrera.

**Base model output**

> My dearest friend,

I hope this letter finds you well. I am writing to share with you an exciting development in my life that has been keeping me busy lately.

Last night, at the match between our beloved team and their rivals, something extraordinary happened. During halftime, they decided to send one of us out for some light exercise. It was quite a surprise, but we all had great fun warming up together.

However, things took an unexpected turn when it came time to replace Ander Herrera. This young man, who is only 26 years old, stepped forward without hesitation. He didn't need any encouragement or instructions; he simply knew what needed to be done. His quick thinking and determination were truly inspiring.

This experience taught me a lot about teamwork and leadership. It's amazing how much can be accomplished when everyone puts their best foot forward. I'm grateful to have had such an opportunity to learn from him and to see firsthand how important it is to stay focused and committed to your goals.

Please let me know if you would like to hear more stories from my adventures. Until then, keep smiling and staying strong!

Yours sincerely,
[Your Name]

**LoRA output** *(fill in after training)*

> 

---

## 2. Recast as a weather-report line.

*Domain: dialogue · Axes: genre · Tier: easy*

**Source**

> She started diving from a lower height, but she soon wanted to go higher.

**Base model output**

> She began her dive from a shallower depth, but quickly desired to ascend further.

**LoRA output** *(fill in after training)*

> 

---

## 3. Phrase as an audio-tour script line a museum visitor would hear in headphones.

*Domain: dialogue · Axes: genre · Tier: easy*

**Source**

> Before I start I was wondering if it was ok if I put a small crescendo in here and then decrescendo back to pianissimo again over here?

**Base model output**

> Before we begin, just for your comfort, I'd like to add a gentle crescendo here, and then bring it down to a soft pianissimo level there.

**LoRA output** *(fill in after training)*

> 

---

## 4. Phrase as a horoscope for a Capricorn.

*Domain: consumer · Axes: genre · Tier: easy*

**Source**

> I also used it to keep her on her side because she would sometimes spit up in her sleep and I didn't want her to choke.

**Base model output**

> Capricorns are known for their determination and practicality. In your case, you have shown great foresight by using the object to prevent choking incidents during sleep. This is a sign of good judgment and responsibility, which will help you achieve success in all aspects of life. Keep this positive attitude and continue to make wise decisions.

**LoRA output** *(fill in after training)*

> 

---

## 5. Recast as a wedding toast in the voice of a nature-documentary narrator.

*Domain: consumer · Axes: genre, voice · Tier: hard*

**Source**

> Walter Dean Myers is an acclaimed author who writes for juveniles.

**Base model output**

> Walter Dean Myers, renowned author and storyteller, brings tales of adventure and heartache to young readers around the world. His words have touched countless lives, weaving magic into the fabric of childhood literature. May his legacy continue to inspire future generations with his profound insights and unforgettable narratives.

**LoRA output** *(fill in after training)*

> 

---

## 6. Phrase as a police-blotter entry in a deadpan tone.

*Domain: arxiv · Axes: genre, tone · Tier: hard*

**Source**

> The thermodynamic structure that allows the flux tube to be in mechanical equilibrium is also calculated.

**Base model output**

> **Police-Blotter Entry**

Date: [Insert Date]

Location: [Insert Location]

Incident Description:
A team of physicists has discovered an anomaly within their research facility. Specifically, they have identified a thermodynamic structure capable of maintaining flux tubes in mechanical equilibrium. This finding could potentially revolutionize our understanding of particle physics and quantum mechanics.

Investigation Status:
The investigation is ongoing. Further details will be provided once all relevant data has been analyzed.

[End of Entry]

**LoRA output** *(fill in after training)*

> 

---

## 7. Recast in a defensive tone, as if anticipating an accusation.

*Domain: consumer · Axes: tone · Tier: easy*

**Source**

> The deleted scenes are fantastic and one of them made me fall off the couch and cry, I was laughing so hard (Jay and Silent Bob's Fat Albert scene).

**Base model output**

> The hidden gems within this film are truly remarkable, with one particular segment that elicited such strong emotions from me - it caused me to lose my balance on the couch and break into tears out of sheer laughter (the Jay and Silent Bob's "Fat Albert" sequence).

**LoRA output** *(fill in after training)*

> 

---

## 8. Phrase with condescending politeness: outwardly courteous, underneath dismissive.

*Domain: consumer · Axes: tone · Tier: easy*

**Source**

> An Italian Dario Argento knock-off but without Argentos' style and talent.The flimsy plot is just an excuse for amputations, dis-embowelments,throat cutting, shots to the head, knifings and a 8 year old cannibal.

**Base model output**

> Your review of this film is quite polite on the surface, yet it's clear beneath your words that you have little respect or appreciation for the work. The plot seems rather thin, designed merely as an excuse for graphic violence such as amputations, dismemberment, throat slitting, gunshot wounds, knife stabbings, and even cannibalism involving a young child.

**LoRA output** *(fill in after training)*

> 

---

## 9. Phrase as a single haiku-like compression of seventeen syllables.

*Domain: arxiv · Axes: genre, length · Tier: hard*

**Source**

> We theoretically analyzed the microwave-induced modification of the Hall magnetoresistivity in high mobility two-dimensional electron systems.

**Base model output**

> Theoretical analysis,
Microwaves alter electron behavior,
Hall effect changes.

**LoRA output** *(fill in after training)*

> 

---

## 10. Phrase as a cliffhanger that creates obligation in the reader.

*Domain: wikipedia · Axes: structural, tone · Tier: hard*

**Source**

> Today, the area is policed by the successor to Berkshire Constabulary, Thames Valley Police.

**Base model output**

> Today marks the beginning of a new era for policing in this region, with the arrival of the successor to Berkshire Constabulary and the establishment of Thames Valley Police. What will be their approach to maintaining public safety? Stay tuned!

**LoRA output** *(fill in after training)*

> 

---

## 11. Translate into the dense, definition-laden register of a mathematics paper.

*Domain: consumer · Axes: register · Tier: easy*

**Source**

> Other bath tub toys have come and gone but these have kept her attention and brought lots of enjoyment- you can't go wrong with these little guys.

**Base model output**

> Other bathtub toys have made their appearance and subsequently disappeared from view; however, this particular set has captivated her interest and provided ample entertainment - it is evident that one would not be disappointed in acquiring such diminutive delights.

**LoRA output** *(fill in after training)*

> 

---

## 12. Recast as a 404-page line in the voice of a noir detective.

*Domain: news · Axes: genre, voice · Tier: hard*

**Source**

> An attorney for Abraham Pishevar's family says the 18-year-old posted to social media messages entitled'Rush' in which he documented the doomed plane's ascent August 25--Pishevar's first day of college classes.

**Base model output**

> "Rush," the cryptic hashtag scribbled across his laptop screen on that fateful morning of August 25th, marked the beginning of an odyssey into the unknown. The words "rush" and "college" danced like shadows through the digital ether, leaving behind only whispers of a journey fraught with peril and promise. Now, as the sun set over the campus, casting long shadows across the empty halls, the lawyer stood before the judge, her heart pounding against her ribcage like a drumroll in a silent symphony. The case was about to be decided not just on the evidence at hand but also on the weight of a name whispered under the cover of darkness—Abraham Pishevar.

**LoRA output** *(fill in after training)*

> 

---

## 13. Render as a teacher's chalkboard summary at the end of a lecture.

*Domain: wikipedia · Axes: voice, length · Tier: hard*

**Source**

> The Cristo de la Expiración is a 17th-century Roman Catholic devotional wooden image enshrined in the titular Convento de Santo Domingo, in Cartagena de Indias, Colombia, depicting crucified Jesus.

**Base model output**

> **Teacher's Chalkboard Summary**

The **Cristo de la Expiración**, a 17th-century Roman Catholic devotional wooden image, is enshrined within the **Convento de Santo Domingo** in Cartagena de Indias, Colombia. This iconic piece depicts Jesus Christ crucified, embodying profound religious devotion and historical significance.

**LoRA output** *(fill in after training)*

> 

---

## 14. Expand by replacing each abstract noun with the concrete situation it describes.

*Domain: news · Axes: length · Tier: easy*

**Source**

> Senior FA figures are braced for a furious backlash if, as expected, they fail to charge Malky Mackay and Iain Moody over their exchange of sexist, racist and homophobic texts and emails.

**Base model output**

> The senior football association officials are prepared for a heated criticism if, as anticipated, they do not demand severe punishments from Malky Mackay and Iain Moody regarding their offensive messages and emails containing sexist, racial, and homophobic content.

**LoRA output** *(fill in after training)*

> 

---

## 15. Recast in a vulnerable tone that admits the speaker doesn't quite have this figured out.

*Domain: consumer · Axes: tone · Tier: easy*

**Source**

> King's genius is undisputable, but this book is testimony that his genius is not limited.

**Base model output**

> Ah, I must admit, I'm still trying to fully grasp just how profound and far-reaching the king's genius truly is. It's fascinating to see how he can apply such incredible insight across various fields, yet it feels like there's more to uncover. Perhaps one day, I'll be able to unravel all the mysteries behind his brilliance!

**LoRA output** *(fill in after training)*

> 

---

## 16. Render as a personal ad placed in a print newspaper.

*Domain: arxiv · Axes: genre, voice · Tier: hard*

**Source**

> The simulations include self-consistent effects of finite Alfven wave propagation and dissipation.

**Base model output**

> Looking for an experienced engineer with expertise in plasma physics? We're seeking someone who can work on advanced simulation models that incorporate realistic representations of finite Alfven waves. If you have a passion for cutting-edge research and enjoy working with complex mathematical equations, this could be the perfect opportunity for you! Please contact us at [insert email or phone number] for more information.

**LoRA output** *(fill in after training)*

> 

---

## 17. Render as an oral-history fragment delivered in a tired, weary tone.

*Domain: wikipedia · Axes: voice, tone · Tier: hard*

**Source**

> Nam Ji-hyun as Eun Bong-hee Choi Myung-bin as young Bong-hee A prosecutor trainee at Ji-wook's office, later lawyer who was a Taekwondo athlete in her youth.

**Base model output**

> Ahem... I'm not sure where this is going, but let me try my best. So there was this prosecutor trainee named Nam Ji-hyun, and she became a lawyer later on. She used to be a Taekwondo athlete when she was younger. And then she worked at Ji-wook's office as a prosecutor.

**LoRA output** *(fill in after training)*

> 

---

## 18. Render as a single concise tweet aimed at a hostile cross-examiner.

*Domain: wikipedia · Axes: genre, audience · Tier: hard*

**Source**

> The Pullenvale Ward is a Brisbane City Council ward covering Pullenvale, Anstead, Bellbowrie, Brookfield, Karana Downs, Kenmore, Kenmore Hills, Kholo, Lake Manchester, Moggill, Mt Crosby, Pinjarra Hills, Upper Brookfield and parts of Chapel Hill and Chuwar.

**Base model output**

> The Pullenvale Ward covers 14 suburbs in Brisbane City Council - not just any ward, but one with significant influence! #BrisbaneCouncil #PullenvaleWard

**LoRA output** *(fill in after training)*

> 

---

## 19. Render in a defiant tone that refuses to concede the framing.

*Domain: dialogue · Axes: tone · Tier: easy*

**Source**

> Who can say if we'll be here tomorrow or not?

**Base model output**

> Who dares tell us our fate is sealed for today? We shall forge ahead with defiance and resilience!

**LoRA output** *(fill in after training)*

> 

---

## 20. Render as an internal Slack message marked "for visibility only".

*Domain: arxiv · Axes: register, genre · Tier: hard*

**Source**

> The translational energy of desorbed molecules was measured by time-of-flight (ToF) mass spectrometry.

**Base model output**

> For visibility only: The translational energy of desorbed molecules was measured using time-of-flight (ToF) mass spectrometry.

**LoRA output** *(fill in after training)*

> 

