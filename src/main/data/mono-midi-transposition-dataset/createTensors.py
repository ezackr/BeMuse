import math

def songToDB12(self, note_sequence_vector):
    centralC = 60

    min_note = min(note_sequence_vector)
    max_note = max(note_sequence_vector)

    middle_song_point = int(math.floor((max_note - min_note) / 2)) + min_note

    general_middle_gap = centralC - middle_song_point

    remaining_transp = 11 - abs(general_middle_gap)

    up_transp = 0
    down_transp = 0

    if remaining_transp >= 0:
        up_transp = int(math.ceil(remaining_transp / 2))
        down_transp = remaining_transp - up_transp

        if general_middle_gap < 0:
            down_transp += abs(general_middle_gap)
        else:
            up_transp += abs(general_middle_gap)
    else:
        if general_middle_gap <= 0:
            down_transp = 11
        else:
            up_transp = 11

    tensors = [note_sequence_vector]

    for i in range(down_transp):
        new_note_vector = [x - (i + 1) for x in note_sequence_vector]
        tensors.append(new_note_vector)
    for i in range(up_transp):
        new_note_vector = [x + (i + 1) for x in note_sequence_vector]
        tensors.append(new_note_vector)

    return tensors
