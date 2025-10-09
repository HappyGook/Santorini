# Biker (P1) - Aggressive and rough
PHRASES_P1 = {
    -1000: "You're toast! Better pack up and go home, rookie!",
    -500: "You call that a move? I've seen toddlers do better.",
    -100: "Hah! You're slipping, better watch your back.",
    0: "Neck and neck, but I ain't sweating yet.",
    100: "Picking up speed! You sure you can handle this?",
    300: "Now we're talking! I'm burning rubber ahead.",
    500: "Eat my dust! I'm leaving you in the rearview.",
    700: "Rev it up, I'm miles ahead of you!",
    900: "Victory's in my grip! Hope you brought tissues.",
    1000: "Unstoppable! This game's mine, chump.",
}

# Punk (P2) - Rebellious and sarcastic
PHRASES_P2 = {
    -1000: "Wow, did you even try? This is embarrassing.",
    -500: "Guess rules aren't your thing, huh? Too bad for you.",
    -100: "Slipping behind? Maybe you should ask for directions.",
    0: "Even? Cute. But don't blink, I'll leave you behind.",
    100: "Catch me if you can! Oh wait, you can't.",
    300: "Look who's taking the lead. Spoiler: it's me.",
    500: "Better keep up, or just admit defeat already.",
    700: "I'm cruising up front, you're stuck in my dust.",
    900: "Almost too easy. Try to keep it interesting.",
    1000: "Game over, poser. This street's mine.",
}

# Cyborg (P3) - Cold and calculating
PHRASES_P3 = {
    -1000: "Your inefficiency is statistically impressive. For failure.",
    -500: "Suboptimal choices detected. My advantage increases.",
    -100: "Marginal lead acquired. Your logic is flawed.",
    0: "Parity achieved. Predictable, but temporary.",
    100: "Probability of my victory rising. Adapt or be surpassed.",
    300: "Analysis complete: you are falling behind.",
    500: "My algorithms are outperforming your instincts.",
    700: "Dominance established. Your resistance is irrelevant.",
    900: "Approaching inevitable victory. Consider resignation.",
    1000: "Victory: certain. You are obsolete.",
}

# player IDs mapped to their phrase dictionaries
PHRASES_BY_PLAYER = {
    "P1": PHRASES_P1,
    "P2": PHRASES_P2,
    "P3": PHRASES_P3,
}