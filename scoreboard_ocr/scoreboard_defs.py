import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
from scoreboard_map import team1, team2, score1, score2, time_clock
from functions import map_leds, ocr_text, process_img

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
            team_match = possible_teams.acronym[team_loc[0]]
            return team_match

    def make_team_acronyms(self):
        self.team_1_tla = translate_team(self.team_1)
        self.team_2_tla = translate_team(self.team_2)

    def create_scoreboard(self):
        # Teams
        t1_c1 = make_led_map(team1['char1'], self.team_1_tla[0])
        t1_c2 = make_led_map(team1['char2'], self.team_1_tla[1])
        t1_c3 = make_led_map(team1['char3'], self.team_1_tla[2])

        t1_c1 = make_led_map(team2['char1'], self.team_2_tla[0])
        t1_c1 = make_led_map(team2['char2'], self.team_2_tla[1])
        t1_c1 = make_led_map(team2['char3'], self.team_2_tla[2])

        # Change digit location depending on length
        if len(self.score_1):
            s1_c1 = make_led_map(score1['dig1_char1'], self.score_1[0])

        if len(self.score_1):
            s1_c1 = make_led_map(score1['dig2_char1'], self.score_1[0])
            s1_c2 = make_led_map(score1['dig2_char2'], self.score_1[1])

        if len(self.score_1):
            s1_c1 = make_led_map(score1['dig3_char1'], self.score_1[0])
            s1_c2 = make_led_map(score1['dig3_char2'], self.score_1[1])
            s1_c3 = make_led_map(score1['dig3_char3'], self.score_1[2])

        # Team 2
        if len(self.score_2):
            s2_c1 = make_led_map(score2['dig1_char1'], self.score_2[0])

        if len(self.score_2):
            s2_c1 = make_led_map(score2['dig2_char1'], self.score_2[0])
            s2_c2 = make_led_map(score2['dig2_char2'], self.score_2[1])

        if len(self.score_2):
            s2_c1 = make_led_map(score2['dig3_char1'], self.score_2[0])
            s2_c2 = make_led_map(score2['dig3_char2'], self.score_2[1])
            s2_c3 = make_led_map(score2['dig3_char3'], self.score_2[2])

        # Time Clock
        t1_c1 = make_led_map(time_clock['clock_dig1'], self.game_clock[0])
        t1_c1 = make_led_map(time_clock['clock_dig2'], self.game_clock[1])
        t1_c1 = make_led_map(time_clock['clock_dig3'], self.game_clock[2])
        t1_c1 = make_led_map(time_clock['clock_dig4'], self.game_clock[3])

    def make_led_map(self, placement, character):
        led_map = dict(zip(placement, character))
        return led_map

    def print_scoreboard(self, mode="simple"):
        if mode == 'simple':
            print(self.team_1+": "+self.score_1+"  -  "+self.team_2+": "+self.score_2" with "+self.play_clock+" remaining.")
        if mode == 'leds':
            led_board = self.led_map

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
