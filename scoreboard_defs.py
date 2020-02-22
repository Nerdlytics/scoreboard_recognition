import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz

class scoreboard:
    """
    Class for handling scoreboard definitions
    Pre-programmed scoreboards since recognition is going so poorly
    Arguments:
        bbox - the bounding box coordinates of the entire scoreboard
        boxes - the scoreboard items within the bounding box, defined by pixel ranges
        sport - the sport of this scoreboard (basketball, soccer, etc.); defines the animations
        channel - the channel of this scoreboard (useful for auto-recognition)
    """
    def __init__(self,
                 bbox,
                 boxes,
                 sport,
                 channel):
        self.bbox = bbox
        self.boxes = boxes
        self.sport = sport
        self.channel = channel

        # Default values
        self.team_1 = ""
        self.team_2 = ""
        self.score_1 = ""
        self.score_2 = ""
        self.game_clock = ""

        # Globals
        self.team_defs = pd.read_csv("team_acronyms.csv", encoding='latin-1')

    def read_scoreboard_screen(self):
        while(True):
            image = np.array(ImageGrab.grab(scoreboard.bbox))

            results = []
            pimg = process_img(image)

            # loop over the bounding boxes
            for (startX, startY, endX, endY) in scoreboard.boxes:
                # extract the actual padded ROI
                roi = pimg[startY:endY, startX:endX]
                text = ocr_text(img=roi, psm="7")
                results.append(((startX, startY, endX, endY), text))

            # Update the attributes
            self.team_1 = results[0][1]
            self.team_2 = results[3][1]
            self.score_1 = results[1][1]
            self.score_2 = results[2][1]
            self.game_clock = results[4][1]

            # Extra clock for some sports
            if self.sport == "basketball":
                self.shot_clock = results[5][1]
            if self.sport == "football":
                self.play_clock = results[5][1]

    def translate_team(self, team_text):
        if len(team_text)<=3:
            return team_text
        else:
            possible_teams = self.team_defs
            ratios = [fuzz.ratio(team_text.lower(), f.lower()) for f in possible_teams.team_name]

            team_loc = [i for i, j in enumerate(ratios) if j == max(ratios)]
            team_match = possible_teams.acronym[team_loc]
            return team_match

# Current pre-programmed scoreboards
premleague_nbc = scoreboard(bbox=(68,75,720,123),
                            boxes=[(70, 0, 162, 48),  # Team 1
                                   (166, 0, 234, 48),  # Score 1
                                   (261, 0, 330, 48),  # Score 2
                                   (334, 0, 432, 48),  # Team 2
                                   (500, 0, 720, 48) # Game Clock
                                  ],
                            sport="soccer",
                            channel="NBCSports"
                            )
