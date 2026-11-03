"""Builds texts.json, the e2e corpus (committed; re-run only to regenerate it).

    .venv/bin/python tests/e2e/kokoro/make_corpus.py

The english-practice checkout comes from the machine-local config (`english_practice_dir` in lokinfer.local.json, or
$LOKINFER_ENGLISH_PRACTICE; python/lokinfer/localconfig.py).

A mix of
- real english-practice texts (read-only, sampled deterministically): the Auto Shadow TTS pack
  (shared_data/tts-th-sounds.expshadata), the shadowing DBs (src/data/db-shadowing-*.json), the phoneme sentences
  and word lists (src/data/phonemes/), the Questions themes (src/data/questions/*.txt);
- texts written here: short phrases, 5-10 s sentences and spoken-style passages, general questions, 20-80 s
  paragraphs (incl. multi-paragraph expository answers) and edge cases (numbers, abbreviations, dates, punctuation,
  quotes, all caps, non-ASCII, line breaks).
Personal and third-party app texts are never used:
lokinfer is public. Their slots hold written texts of the same class / similar length (see `_skip_draws`).

Each entry: id, text, cls (expected length class), source, voice. Voices: ~70 % the app's "Kokoro random" pair
(af_heart / am_echo), the rest spread over 8 more English voices. The app always runs KPipeline with lang_code 'a'
(American G2P), whatever the voice — so the British voices here also get American G2P, exactly like the app.
"""
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Expected length classes (audio duration at speed 1).
CLASSES = {
    "word": "one word (< 1.5 s)",
    "phrase": "short phrase (1-3 s)",
    "sentence": "sentence (2-5 s)",
    "long_sentence": "long sentence (5-10 s)",
    "paragraph": "paragraph (20-40 s, usually one chunk)",
    "long_paragraph": "long paragraph (40-80 s, several chunks)",
    "edge": "edge case (numbers, abbreviations, punctuation, ...)",
}

APP_VOICES = ["af_heart", "am_echo"]
OTHER_VOICES = ["af_bella", "af_nicole", "af_sarah", "am_michael", "am_fenrir", "bf_emma", "bm_george", "bf_isabella"]

PHRASES = [
    "Good morning!", "Thank you so much.", "See you later.", "Not right now.", "Excuse me, sir?",
    "How are you doing?", "That sounds great.", "Let me think about it.", "Over there, on the left.",
    "Wait, really?", "No worries at all.", "Happy birthday!", "What time is it?", "I'm on my way.",
    "Could you repeat that?", "Absolutely not.", "Right, okay then.", "Nice to meet you.",
]

LONG_SENTENCES = [
    "When I finally arrived at the airport, I realized that I had left my passport on the kitchen table at home.",
    "The weather forecast said it would be sunny all weekend, but it has been raining since Friday afternoon.",
    "If you want to improve your pronunciation, you should listen carefully and repeat each sentence out loud.",
    "My neighbor, who works as a nurse at the hospital, usually comes home very late on Thursday nights.",
    "Although the restaurant was quite expensive, the food was absolutely delicious and the service was excellent.",
    "She told me that she would call me back as soon as she found out what had happened to the package.",
    "We spent the whole afternoon walking around the old town, taking photos of the narrow streets and colorful houses.",
    "The new manager explained the project in detail, but most of the team still had a lot of questions afterwards.",
    "Before you leave the office tonight, please make sure that all the windows are closed and the lights are off.",
    "I have been learning to play the guitar for almost three years, and I still can't play my favorite song properly.",
    "Thousands of people gathered in the square to watch the fireworks and celebrate the beginning of the new year.",
    "He said he would rather stay at home and read a good book than go out to a noisy party with strangers.",
    "Honestly, the hardest part of moving to a new country wasn't the language, it was making real friends.",
    "Through the thick fog, the three thieves thought they saw something strange on the other side of the bridge.",
]

PARAGRAPHS = [
    # ~20-40 s
    "Every Saturday morning, my grandmother used to wake up before sunrise to bake bread for the whole family. "
    "The smell would slowly fill the house, and one by one we would wander into the kitchen, still half asleep, "
    "hoping to steal a warm slice before breakfast. She always pretended to be annoyed, but she secretly made an "
    "extra loaf just for us.",
    "Learning a new language is a little bit like building a house. First you need a solid foundation: the basic "
    "grammar and the most common words. Then you add the walls and the roof, which are the expressions and the "
    "patterns you hear every day. Finally, you decorate it with the idioms and the jokes that make it feel like home.",
    "The meeting started twenty minutes late because the projector wasn't working. Someone suggested we just use a "
    "laptop instead, but nobody had the right cable. In the end, we gathered around a single screen, squinting at "
    "tiny charts, and somehow we still managed to make a decision about the budget for next year.",
    "I've never been a morning person. My alarm goes off at seven, and I usually press snooze at least three times. "
    "By the time I actually get out of bed, I have about ten minutes to shower, get dressed and find my keys, which "
    "are never where I think I left them.",
    "Last winter we rented a small cabin in the mountains. There was no internet and barely any phone signal, so we "
    "spent the evenings playing board games by the fire and talking for hours. It was the most relaxing week I've "
    "had in years, and I can't wait to go back.",
    "To make a good cup of tea, start with fresh, cold water and bring it to a boil. Warm the teapot first, then add "
    "one spoon of tea for each person, plus one for the pot. Pour the water, wait three to five minutes, and don't "
    "forget the milk if you like it the British way.",
    "Our team has been working on this product for almost two years. We started with a simple idea: make it easier "
    "for people to practice speaking every day. Since then, we've talked to hundreds of learners, rewritten the app "
    "twice, and learned that the most important feature is simply getting people to open it.",
    "The museum was much bigger than I expected. We spent the first two hours in the ancient history section alone, "
    "looking at pottery, coins and statues that were thousands of years old. By the time we reached the modern art "
    "wing, our feet hurt so much that we just sat on a bench and watched the other visitors.",
    # ~40-80 s, several chunks
    "Last summer my brother and I decided to drive along the whole coast, from the little fishing village where our "
    "grandparents used to live, all the way to the big city in the south. We didn't plan much in advance. Every "
    "morning we looked at the map, picked a direction and simply started driving. We stopped whenever something "
    "caught our eye: a market selling fresh peaches, an old lighthouse on a cliff, a tiny restaurant with only three "
    "tables. In the evenings we looked for cheap hotels near the beach, ate grilled fish, and talked for hours about "
    "our childhood, about the people we used to know, and about the plans we had never dared to talk about. By the "
    "end of the trip we had taken hundreds of photos, but honestly, the best part was simply spending time together "
    "again, without phones ringing or work getting in the way. We have already promised each other to do it again "
    "next year, maybe this time in the other direction.",
    "When people ask me how I learned English, they usually expect me to name a course or an app. The truth is a "
    "bit more boring. I watched the same videos again and again, paused after every sentence, and repeated it out "
    "loud until it sounded right. At first it felt ridiculous. I was talking to my laptop like a parrot. But after a "
    "few weeks I noticed that I could follow conversations much more easily, and that words I had repeated many "
    "times started coming out of my mouth without thinking. That's the whole secret: listen, repeat, and do it "
    "every day, even when you don't feel like it. Ten minutes a day is better than two hours once a week. And if you "
    "record yourself and listen back, you'll hear mistakes you never noticed before, which is uncomfortable, but "
    "it's exactly how you improve. Nobody enjoys hearing their own voice at first, but you get used to it.",
    "The history of the bicycle is surprisingly complicated. The first machines, in the early nineteenth century, "
    "had no pedals at all: riders pushed themselves along with their feet, like children on a balance bike. Later "
    "came the famous high wheelers, with an enormous front wheel and a tiny one at the back. They were fast, but "
    "falling off one was genuinely dangerous, and only brave young men really used them. The so-called safety "
    "bicycle, with two wheels of the same size and a chain driving the rear wheel, changed everything. Suddenly "
    "cycling was something almost anyone could do, including women, who found in the bicycle a new kind of freedom. "
    "Within a few years, cities were full of cyclists, roads were improved, and some historians even argue that the "
    "bicycle helped pave the way, quite literally, for the car.",
    "Good evening everyone, and thank you for coming tonight. I know many of you travelled a long way to be here, "
    "and that means a lot to all of us. When we started this association ten years ago, we were five friends sitting "
    "around a kitchen table with a crazy idea and almost no money. Today we have more than two hundred volunteers, "
    "three community gardens, and a small library that is open every weekend. None of this would have been possible "
    "without you. So before we eat, and before the music starts, I would like to say a special thank you to the "
    "people who work in the shadows: the ones who carry the chairs, who answer the emails at midnight, and who "
    "always show up when it rains. Please raise your glasses. To the next ten years!",
]

EDGE = [
    "It costs $12.99, or about €11.50.",
    "Call me at 555-0123 before 5:30 p.m.",
    "The meeting is on Monday, March 3rd, 2025.",
    "She was born on 07/04/1990 in New York.",
    "We sold 1,234,567 copies in 2023.",
    "The temperature dropped to -5°C last night.",
    "Dr. Smith and Mr. Jones met at St. Mary's Hospital.",
    "The U.S.A. and the U.K. signed the agreement.",
    "NASA, the FBI and the BBC were all there.",
    "I LOVE THIS SONG SO MUCH!",
    "STOP. Don't touch that.",
    "\"Are you sure?\" she asked. \"Absolutely,\" he said.",
    "Wait... what? No way!!!",
    "Well - I mean - it's complicated; isn't it?",
    "Hmm. Okay. Fine. Whatever.",
    "(This part is in parentheses.) And this part isn't.",
    "Café, naïve, résumé, and jalapeño are borrowed words.",
    "We had crème brûlée in São Paulo.",
    "I'd've done it if I could've.",
    "The ratio is 3:2, roughly 66.7%.",
    "Version 2.0.1 was released on 2024-11-15.",
    "Chapter IV begins on page 42.",
    "It's 10 a.m. here, 6 p.m. in Tokyo.",
    "He scored 3/4 on the test, i.e. 75 percent.",
    "Email me at hello@example.com, or visit www.example.com.",
    "First line of the poem.\nSecond line, after a line break.\nAnd the third one.",
    "#1 tip: practice daily & stay calm.",
    "Rock 'n' roll never dies.",
    "OK",
    "A",
    "Hello 👋 world 🌍!",
    "1, 2, 3, 4, 5, 6, 7, 8, 9, 10.",
    "In 1969, Apollo 11 landed on the Moon at 20:17 UTC.",
    "Mississippi.",
    "The quick brown fox jumps over the lazy dog.",
]

# Spoken-style 5-10 s passages (fragments and run-ons, like real speech transcripts).
SPOKEN_PASSAGES = [
    "and then the rain finally stopped. But the river kept rising all night, so by morning the whole lower road was "
    "under water",
    "So then, they had to pack everything up and start again from the beginning. And somehow nobody complained, which "
    "honestly surprised me.",
    "turned out to be the busiest train station in the country, with something like a thousand trains passing through "
    "it every single day, can you imagine that",
    "thing with the little boy and the kite and the old oak tree, and his father climbing up a ladder to get it back "
    "before dinner.",
    "take the story of the lighthouse keeper, for example. He lived alone on that rock for thirty years and wrote a "
    "letter to his sister every week.",
    "like, it changes everything you thought about the place, because the old bridge was actually built in a single "
    "summer by just a few people.",
    "and giving out medals to all of them. A ninety-year-old runner crossing the finish line, basically the whole "
    "town was cheering.",
    "keeping the entire kitchen running. So, as a result, all the bread and the soup had to be ready before the doors "
    "opened at noon.",
]

# General-knowledge questions (9-13 words).
QUESTIONS_WRITTEN = [
    "What does a good night's sleep do for your memory?",
    "Your garden produced twice as many tomatoes this year. How?",
    "What changes when a small town suddenly becomes a tourist destination?",
    "How would you organize a kitchen so that cooking feels effortless?",
    "Last year you moved to a new city. How did that change your routine?",
    "Why measure a city's traffic in travel time rather than in cars?",
]

# Long expository answers (40-80 s, several paragraphs, so several chunks), neutral general topics.
LONG_ANSWERS = [
    "They're layers, from the ground up.\n\n"
    "The soil is the foundation: the minerals from broken rock, the water held between the grains, and the "
    "countless tiny organisms that turn fallen leaves into food for plants. Without healthy soil, nothing else in "
    "the garden works, no matter how much you water or how carefully you choose the seeds. That's why experienced "
    "gardeners spend more time on the ground than on the plants.\n\n"
    "Above that come the roots, which most people never think about. They anchor the plant, store energy for the "
    "winter, and trade sugar with fungi in exchange for water and nutrients from far away.\n\n"
    "Then there are the stems and leaves, the part we actually see. Leaves capture sunlight and turn it into sugar, "
    "while the stems carry water up and food down, like a slow two-way road. A tall tree moves hundreds of liters "
    "of water every day this way, silently.\n\n"
    "Finally, flowers and fruit sit on top. They exist mostly to attract insects and animals that carry pollen and "
    "seeds somewhere new. Each layer depends on the one below it, which is why a gardener who fixes the soil first "
    "usually fixes everything else too.",

    "The usual advice, and what I'd recommend: plan the week on Sunday evening, write down the three things that "
    "really matter, and protect the time for them before anything else gets scheduled. Most people do the opposite. "
    "They fill their calendar with meetings and errands, and then try to squeeze the important work into whatever "
    "gaps are left, which almost never works for very long.\n\n"
    "The second piece is to batch the small tasks. Answering messages, paying bills and making phone calls all feel "
    "quick, but switching between them costs a lot of attention. Doing them together, once or twice a day, keeps the rest of "
    "the day free for deeper work that needs a clear head.\n\n"
    "Third, leave some empty space. A schedule with no slack breaks the moment anything unexpected happens, and "
    "something unexpected always happens. A free hour in the afternoon is not wasted time; it's insurance.\n\n"
    "And finally, review on Friday. Ten minutes is enough to see what worked, what didn't, and what to move to "
    "next week without feeling guilty. Over a few months, those short reviews show you patterns: which days are "
    "always overloaded, which tasks you keep postponing, and which habits really make a difference.",

    "Because bread is mostly air, not flour. To get a light loaf, you need two things working together: gas "
    "produced by the yeast, and a dough strong enough to trap it.\n\n"
    "The yeast eats the sugars in the flour and releases carbon dioxide. At the same time, kneading lines up the "
    "proteins in the flour into long, stretchy strands called gluten. These strands form a kind of net, and as the "
    "gas is produced, the net stretches into thousands of small bubbles. If the dough is too weak, the bubbles "
    "burst and the loaf comes out flat and heavy.\n\n"
    "Time matters as much as technique. A slow rise in a cool kitchen gives the yeast and bacteria longer to "
    "develop flavor, which is why many bakers leave their dough overnight.\n\n"
    "Then comes the oven. The heat makes the gas expand one last time, the water turns to steam, and the outside "
    "dries into a crust. That final burst is what bakers call oven spring, and it's the moment everyone waits for. A few minutes later, the kitchen smells like a bakery, and the hardest part "
    "is waiting for the loaf to cool down.",

    "The first thing to understand is that a river is never really the same from one day to the next. The water "
    "you see this morning will be in the sea within a week, replaced by rain that fell on distant hills.\n\n"
    "Rivers also shape the land around them. On the outside of every bend the current is faster, so it eats into "
    "the bank, while on the inside it slows down and leaves sand behind.\n\n"
    "Over hundreds of years, this slowly pushes the bends further and further outward, until one day a flood cuts "
    "straight across the narrow neck of land between two loops. The old loop is left behind as a curved lake, "
    "sometimes called an oxbow lake, and the river simply carries on along its new, shorter path. If you look at "
    "an aerial photo of a wide valley, you can often spot several of these old loops at once.\n\n"
    "That's why maps of rivers go out of date, and why towns built on their banks keep moving their bridges. A river may look calm and permanent, but on the scale of centuries it behaves more "
    "like a slowly swinging rope.",

    "The short answer is that it saves an enormous amount of energy over a whole year. A well-insulated house "
    "keeps the heat inside during winter and outside during summer, so the heating and cooling systems run much "
    "less often, and the bills drop accordingly.\n\n"
    "Most heat escapes through a few predictable places: the roof, the windows, the gaps around doors, and the "
    "walls. Warm air rises, so an uninsulated attic is usually the biggest leak. Adding a thick layer of insulation "
    "there is often the cheapest improvement with the fastest payback, sometimes within two or three winters.\n\n"
    "Windows come next. Double glazing traps a layer of air or gas between two panes, which slows the flow of heat "
    "considerably. Older single panes can lose ten times more heat than a good modern window, and they also let in "
    "street noise.\n\n"
    "Then there are the small things: draft stoppers, thick curtains, and sealing the cracks around pipes. Each "
    "one seems minor, but together they add up to a much more comfortable home. They are also cheap enough to do in a single weekend, with no special tools and "
    "no help from a professional.",

    "Far more than people expect. Often more than the trip itself.\n\n"
    "On a long journey, the planning is where most of the real decisions happen. You choose the route, the season, "
    "the budget and the pace, and each of those choices quietly shapes everything that follows. Traveling in the "
    "rainy season might save half the cost of the hotels, for example, but it also means some roads are closed and "
    "some islands are simply impossible to reach. A slower pace means fewer places, but much deeper memories of "
    "each one.\n\n"
    "Then there's the paperwork. Visas, vaccinations, insurance and bookings all take time, and some of them need "
    "to be done months in advance. Forgetting one small document can turn a smooth arrival into a long, stressful "
    "afternoon at a border office, with no phone signal and nowhere to sit. It happens to experienced travelers "
    "far more often than they like to admit.\n\n"
    "That said, the best trips always leave room for surprises. Plan the skeleton carefully, and let the days "
    "themselves fill in the rest. Some of the best memories come from the bus you missed or the café you found by accident.",

    "I go from the big picture down to the smallest detail.\n\n"
    "First, I read the whole recipe from start to finish before touching anything. That tells me how long it will "
    "really take, which steps can happen at the same time, and whether I need to start something the night "
    "before. Nothing is worse than discovering, halfway through, that the meat should have been marinating for "
    "twelve hours.\n\n"
    "Second, I prepare every ingredient in advance. Vegetables are chopped, spices are measured, and the pans are "
    "out on the counter. Professional cooks call this mise en place, and it's the main reason restaurant kitchens "
    "can cook so quickly under pressure.\n\n"
    "Third, I cook in the order of the slowest dish. Rice and sauces go on first, quick things like fish or "
    "greens go on last, so everything is ready together.\n\n"
    "Finally, I taste at every stage. Salt, acid and heat are easy to adjust early and almost impossible to fix "
    "once the dish is on the plate. And when it goes wrong anyway, I write down what happened, so the next attempt "
    "is a little better than the last.",

    "When the weather doesn't follow the usual pattern, forecasters rely more on computer models than on "
    "experience.\n\n"
    "These models divide the atmosphere into millions of small boxes, each a few kilometers wide, and calculate "
    "how temperature, pressure, wind and moisture will change in each one over the next few minutes. Then they "
    "repeat the calculation again and again, step by step, until they reach several days into the future. That is a staggering amount of arithmetic, and it runs on some of the "
    "largest computers in the world.\n\n"
    "The difficulty is that small errors grow quickly. A tiny mistake in the starting temperature over the ocean "
    "can become a completely different storm a week later. To deal with this, forecasters run the same model "
    "dozens of times with slightly different starting conditions, and look at how much the results disagree.\n\n"
    "If all the runs agree, the forecast is confident. If they spread out in every direction, the honest answer is "
    "that nobody really knows yet, and that's exactly what a good forecast should say. That's also why forecasts "
    "for tomorrow are usually reliable, while forecasts for ten days ahead are little more than educated guesses.",

    "A public library is much more than a building full of books. It's one of the few places in a modern city "
    "where anyone can walk in, stay as long as they like, and never be asked to buy anything. That simple fact "
    "makes it valuable to students, retired people, job seekers and families alike.\n\n"
    "Inside, the collection has changed a lot over the last twenty years. Besides books, most libraries now lend "
    "films, music, audiobooks, and sometimes even tools, musical instruments or board games. Many offer free "
    "computers and internet access, which matters enormously for people who don't have either at home. Librarians "
    "spend a growing part of their day helping visitors fill in online forms, write a resume, or find reliable "
    "information.\n\n"
    "Libraries also host events: reading groups, language classes, homework clubs and small concerts. Children's story time on Saturday mornings is often the busiest hour of the whole week.\n\n"
    "And on a cold winter afternoon, they're simply a warm, quiet place to sit and think. For many people living alone, the friendly hello at the front desk is "
    "the only conversation they have all day, and that alone makes the library worth protecting.",

    "Composting works best at home, so where you put the bin decides whether you'll actually keep using it. A "
    "spot that is too far from the kitchen means scraps pile up in the sink, and a spot in full sun means the pile "
    "dries out and stops breaking down. Ideally, it sits in partial shade, on bare soil, a short walk from the "
    "back door.\n\n"
    "The recipe itself is simple. You mix green material, like vegetable peelings, coffee grounds and fresh grass, "
    "with brown material, like dry leaves, cardboard and straw. The greens provide nitrogen, the browns provide "
    "carbon, and together they feed the bacteria and worms that do the real work. A rough balance of one part "
    "green to two parts brown is a good start.\n\n"
    "Turning the pile every week or two brings in air and speeds everything up. Without air, the pile starts to "
    "smell, which is usually the first sign that something is off.\n\n"
    "After a few months, the bottom of the bin turns dark and crumbly, and smells like a forest floor. That's "
    "finished compost, ready to spread on the garden beds in spring, where it feeds next year's vegetables.",
]


def main():
    argv = sys.argv[1:]
    from lokinfer import localconfig
    app = Path(localconfig.require("english_practice_dir"))
    rng = random.Random(20261103)
    items = []

    def add(text, cls, source):
        items.append({"text": text, "cls": cls, "source": source})

    def jl(rel):
        return json.loads((app / rel).read_text())

    def pick(seq, n):
        return rng.sample(list(seq), n)

    def _skip_draws(population, n):
        # The corpus once sampled n entries from a source of this size here (sources dropped since, see the
        # docstring); the same draws keep the rest of the corpus (later samples, the voice shuffle) unchanged.
        rng.sample(range(population), n)

    # single words (phoneme word lists) and short phrases
    words = [w["word"] for lst in jl("src/data/phonemes/word-lists.json").values() for w in lst]
    for w in pick(sorted(set(words)), 22):
        add(w, "word", "app:phonemes/word-lists")
    for p in PHRASES:
        add(p, "phrase", "written")
    # real app sentences
    for e in pick(jl("shared_data/tts-th-sounds.expshadata/data.json")["entries"], 30):
        add(e["text"], "sentence", "app:tts-th-sounds pack")
    for level, n in (("easy", 10), ("medium", 14), ("hard", 14)):
        for s in pick(jl(f"src/data/db-shadowing-{level}.json"), n):
            add(s, "sentence" if level != "hard" else "long_sentence", f"app:db-shadowing-{level}")
    for s in pick([s for lst in jl("src/data/phonemes/sentences.json").values() for s in lst], 14):
        add(s, "sentence", "app:phonemes/sentences")
    for theme, n in (("about-myself", 3), ("phone-transactions", 3), ("travel-destinations", 2), ("computer-science", 2),
                     ("explain-how-it-works", 2), ("pitch-an-idea", 2), ("describe-with-emotions", 2),
                     ("roleplay-character", 2)):
        lines = [l.strip() for l in (app / f"src/data/questions/{theme}.txt").read_text().splitlines() if l.strip()]
        for s in pick(lines, n):
            add(s, "sentence", f"app:questions/{theme}")
    _skip_draws(500, 8)
    for s in SPOKEN_PASSAGES:
        add(s, "long_sentence", "written")
    for s in LONG_SENTENCES:
        add(s, "long_sentence", "written")
    _skip_draws(471, 6)
    for q in QUESTIONS_WRITTEN:
        add(q, "sentence", "written")
    _skip_draws(471, 10)
    for a in LONG_ANSWERS:
        add(a, "long_paragraph", "written")
    for i, p in enumerate(PARAGRAPHS):
        add(p, "paragraph" if i < 8 else "long_paragraph", "written")
    for s in EDGE:
        add(s, "edge", "written")

    # voices: ~70 % the app pair (alternating), the rest round-robin over the others, in a shuffled order
    order = list(range(len(items)))
    rng.shuffle(order)
    other = app_n = 0
    for rank, i in enumerate(order):
        if rank % 10 < 7:
            items[i]["voice"] = APP_VOICES[app_n % 2]
            app_n += 1
        else:
            items[i]["voice"] = OTHER_VOICES[other % len(OTHER_VOICES)]
            other += 1
    for i, it in enumerate(items):
        it["id"] = f"t{i:03d}"
    out = {"classes": CLASSES, "voices": {"app": APP_VOICES, "other": OTHER_VOICES},
           "texts": [{k: it[k] for k in ("id", "cls", "voice", "source", "text")} for it in items]}
    (HERE / "texts.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    from collections import Counter
    print(len(items), "texts;", dict(Counter(it["cls"] for it in items)), dict(Counter(it["voice"] for it in items)))


if __name__ == "__main__":
    main()
