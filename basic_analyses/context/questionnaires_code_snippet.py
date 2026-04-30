# =============================================================================
# INDIVIDUAL QUESTIONNAIRES
# =============================================================================
def _tellegen(subject_number: str, win: pg.Surface) -> list[str]:
    """Run the Tellegen Absorption Scale (4-point Likert version)."""

    scale = [
        '0 - Never',
        '1 - Rarely',
        '2 - Often',
        '3 - Always'
    ]

    questions = [
        ('Sometimes I feel and experience things as I did when I was a child.', scale),
        ('I can be greatly moved by eloquent or poetic language.', scale),
        ('While watching a movie, a TV show, or a play, I may sometimes become so involved that I forget about myself and my surroundings and experience the story as if it were real and as if I were taking part in it.', scale),
        ('If I stare at a picture and then look away from it, I can sometimes "see" an image of the picture, almost as if I were still looking at it.', scale),
        ('Sometimes I feel as if my mind could envelop the whole earth.', scale),
        ('I like to watch cloud shapes in the sky.', scale),
        ('If I wish, I can imagine (or daydream) some things so vividly that they hold my attention as a good movie or a story does.', scale),
        ('I think I really know what some people mean when they talk about mystical experiences.', scale),
        ('I sometimes "step outside" my usual self and experience an entirely different state of being.', scale),
        ('Textures such as wool, sand, wood sometimes remind me of colors or music.', scale),
        ('Sometimes I experience things as if they were doubly real.', scale),
        ("When I listen to music, I can get so caught up in it that I don't notice anything else.", scale),
        ('If I wish, I can imagine that my whole body is so heavy that I could not move it if I wanted to.', scale),
        ('I can often somehow sense the presence of another person before I actually see or hear him/her.', scale),
        ('The crackle and flames of a woodfire stimulate my imagination.', scale),
        ('It is sometimes possible for me to be completely immersed in nature or art and to feel as if my whole state of consciousness has somehow been temporarily altered.', scale),
        ('Different colors have distinctive and special meanings to me.', scale),
        ('I am able to wander off into my own thought while doing a routine task and actually forget that I am doing the task, and then find a few minutes later that I have completed it.', scale),
        ('I can sometimes recollect certain past experiences in my life with such clarity and vividness that it is like living them again or almost so.', scale),
        ('Things that might seem meaningless to others often make sense to me.', scale),
        ('While acting in a play, I think I would really feel the emotions of the character and "become" him/her for the time being, forgetting both myself and the audience.', scale),
        ('My thoughts often do not occur as words but as visual images.', scale),
        ('I often take delight in small things (like the five pointed star shape that appears when you cut an apple across the core or the colors in soap bubbles).', scale),
        ('When listening to organ music or other powerful music I sometimes feel as if I am being lifted into the air.', scale),
        ('Sometimes I can change noise into music by the way I listen to it.', scale),
        ('Some of my most vivid memories are called up by scents and smells.', scale),
        ('Certain pieces of music remind me of pictures or moving patterns of color.', scale),
        ('I often know what someone is going to say before he or she says it.', scale),
        ('I often have "physical memories"; for example, after I have been swimming I may still feel as if I am in the water.', scale),
        ('The sound of a voice can be so fascinating to me that I can just go on listening to it.', scale),
        ('At times I somehow feel the presence of someone who is not physically there.', scale),
        ('Sometimes thoughts and images come to me without the slightest effort on my part.', scale),
        ('I find that different odors have different colors.', scale),
        ('I can be deeply moved by a sunset.', scale),
    ]

    return _run_questionnaire(win, subject_number, 'tellegen', questions, intro_text=tellegenScaleIntro, extract_numeric=True)




def _vhq(subject_number: str, win: pg.Surface) -> list[str]:
    """Run the Hearing Voices Questionnaire (Posey & Losch, 1983)."""

    scale = ['Yes', 'No']

    questions = [
        ("Sometimes I have thought I heard people say my name... like in a store when you walk past some people you don't know... but I know they didn't really say my name so I just go on.\n\nHas something like this ever happened to you?", scale),
        ("Sometimes when I am just about to fall asleep, I hear my name as if spoken aloud.\n\nHappened to you?", scale),
        ("When I wake up in the morning... but stay in bed for a few minutes, sometimes I hear my mother's voice... when she's not there. Like now when I'm living in the dorm. What I hear is her voice saying stuff like, 'Now come on and get up' or 'Don't be late for school.' I'm used to it and it doesn't bother me.\n\nHas a similar experience happened to you?", scale),
        ("I hear a voice that is kind of garbled... can't really tell what it says... sometimes just as I go to sleep.\n\nHappened to you?", scale),
        ("I've had experiences of hearing something just when going asleep or waking up.\n\nHave you had any experience with hearing something just when going asleep or waking up?", scale),
        ("When I was little, I had an imaginary playmate. I remember that I really thought I heard her voice when we talked. That went away... hearing her voice... but for awhile it was just like a real voice.\n\nDid you have an imaginary playmate and hear his/her voice aloud?", scale),
        ("Every now and then — not real often — I think I hear my name on the radio.\n\nHappened to you?", scale),
        ("Sometimes when I'm in the house all alone, I hear a voice call my name. No, it really isn't scary. It was at first, but not now... it's just once... like 'Sally'... kind of quick and like somebody's calling me. I guess I kind of know that it really isn't somebody and it's really me... but it does sound like a real voice.\n\nHappened to you?", scale),
        ("Last summer I was hanging-up clothes in the backyard. Suddenly I heard my husband call my name from inside the house. He sounded like something was wrong and was loud and clear. I ran in... but he was out in the garage and hadn't called at all. Obviously I guess I made it up... but it sounded like a real voice and it was my husband's.\n\nThis or something similar happen to you?", scale),
        ("I've heard the doorbell or the phone ring when it didn't.\n\nHappen to you?", scale),
        ("I hear my thoughts aloud.\n\nHappen to you?", scale),
        ("I have heard God's voice... not that he made me know in my heart... but as a real voice.\n\nHappen to you?", scale),
        ("I drive a lot at night. My job has a lot of travel to it. Sometimes late at night, when I'm tired, I hear sounds in the backseat like people talking... but I can't tell what they say... just a word here and there. When this first started happening... when I first started driving at night so much... four or five years ago... it scared the hell out of me. But now I'm used to it. I think I do it because I'm tired and by myself.\n\nAnything similar happen to you?", scale),
        ("Almost every morning while I do my housework, I have a pleasant conversation with my dead grandmother. I talk to her and quite regularly hear her voice actually aloud.\n\nAnything similar happen to you?", scale),
    ]

    return _run_questionnaire(win, subject_number, 'vhq', questions, intro_text=vhqIntro)


def _launay_slade(subject_number: str, win: pg.Surface) -> list[str]:
    """Run the Launay-Slade Hallucination Scale – Extended (16 Likert questions)."""
    scale = ['0 - Certainly does not apply to me', 
             '1 - Possibly does not apply to me', 
             '2 - Unsure',
             '3 - Possibly applies to me',
             '4 - Certainly applies to me'
            ]
    
    questions = [
        ('Sometimes a passing thought will seem so real that it frightens me.', scale),
        ('Sometimes my thoughts seem as real as actual events in my life.', scale),
        ('No matter how much I try to concentrate on my work unrelated thoughts always creep into my mind.', scale),
        ("In the past I have had the experience of hearing a person's voice and then found that there was no one there.", scale),
        ('The sounds I hear in my daydreams are generally clear and distinct.', scale),
        ('The people in my daydreams seem so true to life that I sometimes think they are.', scale),
        ('In my daydreams I can hear the sound of a tune almost as clearly as if I were actually listening to it.', scale),
        ('I often hear a voice speaking my thoughts aloud.', scale),
        ('I have been troubled by hearing voices in my head.', scale),
        ("On occasions I have seen a person's face in front of me when no-one was in fact there.", scale),
        ('Sometimes, immediately prior to falling asleep or upon awakening, I have had the experience of having seen, felt or heard something or someone that wasn’t there, or I had the feeling of being touched even though no one was there.', scale),
        ('Sometimes, immediately prior to falling asleep or upon awakening, I have felt that I was floating or falling, or that I was leaving my body temporarily.', scale),
        ('On certain occasions I have felt the presence of someone close who had passed away.', scale),
        ('In the past, I have smelt a particular odour even though there was nothing there.', scale),
        ("I have had the feeling of touching something or being touched and then found that nothing or no one was there.", scale),
        ("Sometimes, I have seen objects or animals even though there was nothing there.", scale),
    ]
    
    intro = launeyScaleIntro
    return _run_questionnaire(win, subject_number, 'launay_slade', questions, intro_text=intro, extract_numeric=True)


def _dissociative_experiences(subject_number: str, win: pg.Surface) -> list[str]:
    """Run the Dissociative Experiences Scale (28 questions)."""
    scale = ['0% (Never)', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100% (Always)']
    
    questions = [
        ("Some people have the experience of driving a car and suddenly realizing that they don't remember what has happened during all or part of the trip. Select a box to show what percentage of the time this happens to you.", scale),
        ('Some people find that sometimes they are listening to someone talk and they suddenly realize that they did not hear all or part of what was said. Select a box to show what percentage of the time this happens to you.', scale),
        ('Some people have the experience of finding themselves in a place and having no idea how they got there. Select a box to show what percentage of the time this happens to you.', scale),
        ("Some people have the experience of finding themselves dressed in clothes that they don't remember putting on. Select a box to show what percentage of the time this happens to you.", scale),
        ('Some people have the experience of finding new things among their belongings that they do not remember buying. Select a box to show what percentage of the time this happens to you.', scale),
        ('Some people sometimes find that they are approached by people that they do not know who call them by another name or insist that they have met them before. Select a box to show what percentage of the time this happens to you.', scale),
        ('Some people sometimes have the experience of feeling as though they are standing next to themselves or watching themselves do something as if they were looking at another person. Select a box to show what percentage of the time this happens to you.', scale),
        ('Some people are told that they sometimes do not recognize friends or family members. Select a box to show what percentage of the time this happens to you.', scale),
        ('Some people find that they have no memory for some important events in their lives (for example, a wedding or graduation). Select a box to show what percentage of the time this happens to you.', scale),
        ('Some people have the experience of being accused of lying when they do not think that they have lied. Select a box to show what percentage of the time this happens to you.', scale),
        ('Some people have the experience of looking in a mirror and not recognizing themselves. Select a box to show what percentage of the time this happens to you.', scale),
        ('Some people sometimes have the experience of feeling that other people, objects, and the world around them are not real. Select a box to show what percentage of the time this happens to you.', scale),
        ('Some people sometimes have the experience of feeling that their body does not seem to belong to them. Select a box to show what percentage of the time this happens to you.', scale),
        ('Some people have the experience of sometimes remembering a past event so vividly that they feel as if they were reliving that event. Select a box to show what percentage of the time this happens to you.', scale),
        ('Some people have the experience of not being sure whether things that they remember happening really did happen or whether they just dreamed them. Select a box to show what percentage of the time this happens to you.', scale),
        ('Some people have the experience of being in a familiar place but finding it strange and unfamiliar. Select a box to show what percentage of the time this happens to you.', scale),
        ('Some people find that when they are watching television or a movie they become so absorbed in the story that they are unaware of other events happening around them. Select a box to show what percentage of the time this happens to you.', scale),
        ('Some people sometimes find that they become so involved in a fantasy or daydream that it feels as though it were really happening to them. Select a box to show what percentage of the time this happens to you.', scale),
        ('Some people find that they sometimes are able to ignore pain. Select a box to show what percentage of the time this happens to you.', scale),
        ('Some people find that they sometimes sit staring off into space, thinking of nothing, and are not aware of the passage of time. Select a box to show what percentage of the time this happens to you.', scale),
        ('Some people sometimes find that when they are alone they talk out loud to themselves. Select a box to show what percentage of the time this happens to you.', scale),
        ('Some people find that in one situation they may act so differently compared with another situation that they feel almost as if they were two different people. Select a box to show what percentage of the time this happens to you.', scale),
        ('Some people sometimes find that in certain situations they are able to do things with amazing ease and spontaneity that would usually be difficult for them (for example, sports, work, social situations, etc.). Select a box to show what percentage of the time this happens to you.', scale),
        ('Some people sometimes find that they cannot remember whether they have done something or have just thought about doing that thing (for example, not knowing whether they have just mailed a letter or have just thought about mailing it). Select a box to show what percentage of the time this happens to you.', scale),
        ('Some people find evidence that they have done things that they do not remember doing. Select a box to show what percentage of the time this happens to you.', scale),
        ('Some people sometimes find writings, drawings, or notes among their belongings that they must have done but cannot remember doing. Select a box to show what percentage of the time this happens to you.', scale),
        ('Some people sometimes find that they hear voices inside their head that tell them to do things or comment on things that they are doing. Select a box to show what percentage of the time this happens to you.', scale),
        ('Some people sometimes feel as if they are looking at the world through a fog so that people and objects appear far away or unclear. Select a box to show what percentage of the time this happens to you.', scale),
    ]
    
    intro = dissociativeExperiencesIntro
    return _run_questionnaire(win, subject_number, 'dissociative_experiences', questions, intro_text=intro, extract_numeric=True)


def _flow_state_scale(subject_number: str, win: pg.Surface) -> list[str]:
    """Run the Short Flow State Scale (S FSS-2)."""

    scale = [
        '1 - Strongly Disagree',
        '2 - Disagree',
        '3 - Neither Agree nor Disagree',
        '4 - Agree',
        '5 - Strongly Agree'
    ]

    questions = [
        ('I felt I was competent enough to meet the demands of the situation.', scale),
        ('I did things spontaneously and automatically without having to think.', scale),
        ('I had a strong sense of what I wanted to do.', scale),
        ('I had a good idea about how well I was doing while I was involved in the task/activity.', scale),
        ('I was completely focused on the task at hand.', scale),
        ('I had a feeling of total control over what I was doing.', scale),
        ('I was not worried about what others may have been thinking of me.', scale),
        ('The way time passed seemed to be different from normal.', scale),
        ('I found the experience extremely rewarding.', scale),
    ]

    intro = flowStateIntro

    return _run_questionnaire(
        win,
        subject_number,
        'flow_state_scale',
        questions,
        intro_text=intro,
        extract_numeric=True
    )



def _bais_v(subject_number: str, win: pg.Surface) -> list[str]:
    """Run BAIS-V (Bucknell Auditory Imagery Scale – Vividness)."""

    scale = [
        '1 - No Image Present at All',
        '2',
        '3',
        '4 - Fairly Vivid',
        '5',
        '6',
        '7 - As Vivid As the Actual Sound'
    ]

    questions = [
        ('For this item, consider the beginning of the song "Happy Birthday." Imagine hearing a trumpet beginning the piece.', scale),
        ('For this item, consider ordering something over the phone. Imagine hearing the voice of an elderly clerk assisting you.', scale),
        ('For this item, consider being at the beach. Imagine the sound of the waves crashing against nearby rocks.', scale),
        ('For this item, consider going to a dentist appointment. Imagine hearing the loud sound of the dentist\'s drill.', scale),
        ('For this item, consider being present at a jazz club. Imagine hearing a saxophone solo.', scale),
        ('For this item, consider being at a live baseball game. Imagine the cheer of the crowd as a player hits the ball.', scale),
        ('For this item, consider attending a choir rehearsal. Imagine hearing an all-children\'s choir singing the first verse of a song.', scale),
        ('For this item, consider attending an orchestral performance of Beethoven\'s Fifth. Imagine the sound of the ensemble playing.', scale),
        ('For this item, consider listening to a rain storm. Imagine hearing gentle rain.', scale),
        ('For this item, consider attending classes. Imagine hearing the slow-paced voice of your English teacher.', scale),
        ('For this item, consider seeing a live opera performance. Imagine hearing the voice of an opera singer in the middle of a verse.', scale),
        ('For this item, consider attending a new tap-dance performance. Imagine the sound of tap-shoes on the stage.', scale),
        ('For this item, consider a kindergarten class. Imagine hearing the voice of the teacher reading a story to the children.', scale),
        ('For this item, consider driving in a car. Imagine hearing an upbeat rock song on the radio.', scale),
    ]

    intro = baisVIntro

    return _run_questionnaire(win, subject_number, 'bais_v', questions, intro_text=intro, extract_numeric=True)



def _bais_c(subject_number: str, win: pg.Surface) -> list[str]:
    """Run BAIS-C (Bucknell Auditory Imagery Scale – Control)."""

    scale = [
        '1 - No Image Present at All',
        '2',
        '3',
        '4 - Could Change the Image but With Effort',
        '5',
        '6',
        '7 - Extremely Easy to Change the Image'
    ]

    questions = [
        ('For this pair, consider the beginning of the song "Happy Birthday."\n\n'
         'a. The sound of a trumpet beginning the piece.\n'
         'b. The trumpet stops and a violin continues the piece.',
         scale),

        ('For this pair, consider ordering something over the phone.\n\n'
         'a. The voice of an elderly clerk assisting you.\n'
         'b. The elderly clerk leaves and the voice of a younger clerk is now on the line.',
         scale),

        ('For this pair, consider being at the beach.\n\n'
         'a. The sound of the waves crashing against nearby rocks.\n'
         'b. The waves are now drowned out by the loud sound of a boat\'s horn out at sea.',
         scale),

        ('For this pair, consider going to a dentist appointment.\n\n'
         'a. The loud sound of the dentist\'s drill.\n'
         'b. The drill stops and you can now hear the soothing voice of the receptionist.',
         scale),

        ('For this pair, consider being present at a jazz club.\n\n'
         'a. The sound of a saxophone solo.\n'
         'b. The saxophone is now accompanied by a piano.',
         scale),

        ('For this pair, consider being at a live baseball game.\n\n'
         'a. The cheer of the crowd as a player hits the ball.\n'
         'b. Now the crowd boos as the fielder catches the ball.',
         scale),

        ('For this pair, consider attending a choir rehearsal.\n\n'
         'a. The sound of an all-children\'s choir singing the first verse of a song.\n'
         'b. An all-adults\' choir now sings the second verse of the song.',
         scale),

        ('For this pair, consider attending an orchestral performance of Beethoven\'s Fifth.\n\n'
         'a. The sound of the ensemble playing.\n'
         'b. The ensemble stops but the sound of a piano solo is present.',
         scale),

        ('For this pair, consider listening to a rain storm.\n\n'
         'a. The sound of gentle rain.\n'
         'b. The gentle rain turns into a violent thunderstorm.',
         scale),

        ('For this pair, consider attending classes.\n\n'
         'a. The slow-paced voice of your English teacher.\n'
         'b. The pace of the teacher\'s voice gets faster at the end of class.',
         scale),

        ('For this pair, consider seeing a live opera performance.\n\n'
         'a. The voice of an opera singer in the middle of a verse.\n'
         'b. The opera singer now reaches the end of the piece and holds the final note.',
         scale),

        ('For this pair, consider attending a new tap-dance performance.\n\n'
         'a. The sound of tap-shoes on the stage.\n'
         'b. The sound of the shoes speeds up and gets louder.',
         scale),

        ('For this pair, consider a kindergarten class.\n\n'
         'a. The voice of the teacher reading a story to the children.\n'
         'b. The teacher stops reading for a minute to talk to another teacher.',
         scale),

        ('For this pair, consider driving in a car.\n\n'
         'a. The sound of an upbeat rock song on the radio.\n'
         'b. The song is now masked by the sound of the car coming to a screeching halt.',
         scale),
    ]

    intro = baisCIntro

    return _run_questionnaire(win, subject_number, 'bais_c', questions, intro_text=intro, extract_numeric=True)



def stanford_sleepiness_scale(subject_number: str, win: pg.Surface) -> str:
    """Run Stanford Sleepiness Scale (SSS)."""

    pg.mouse.set_visible(True)

    options = [
        '1 - Feeling active, vital, alert, or wide awake',
        '2 - Functioning at high levels, but not at peak; able to concentrate',
        '3 - Awake, but relaxed; responsive but not fully alert',
        '4 - Somewhat foggy, let down',
        '5 - Foggy; losing interest in remaining awake; slowed down',
        '6 - Sleepy, woozy, fighting sleep; prefer to lie down',
        '7 - No longer fighting sleep, sleep onset soon; having dream-like thoughts'
    ]

    response = _run_single_question(
        win,
        'Please select the statement that best describes your current level of sleepiness:',
        options,
        'stanford_sleepiness'
    )

    return response

