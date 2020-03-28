from scoreboard_ocr.scoreboard_defs import scoreboard, premleague_nbc
premleague_nbc.team_1 = 'Chelsea'
premleague_nbc.team_2 = 'Liverpool'
premleague_nbc.score_1 = '5'
premleague_nbc.score_2 = '0'
premleague_nbc.game_clock = '35:35'
premleague_nbc.make_team_acronyms()
premleague_nbc.create_scoreboard_maps()
premleague_nbc.make_scoreboard()

from scoreboard_ocr.scoreboard_display import scoreboard_display
scoreboard_done = []
for row in scoreboard_display:
    if any(int(x) in sb for x in row):
        newrow = [str(sb[int(k)]) for k in row if int(k) in sb]
        newrow = ['.' if x=='0' else 'O' for x in newrow]
        scoreboard_done.append(newrow)
    else:
        scoreboard_done.append(['.','.','.','.','.','.','.','.','.','.',
                                '.','.','.','.','.','.','.','.','.','.',
                                '.','.','.','.','.','.'])
print('\n'.join([''.join(['{:4}'.format(item) for item in row])
                             for row in scoreboard_done]))