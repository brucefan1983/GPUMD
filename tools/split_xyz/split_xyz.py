import io as _io
import sys as _sys
import random as _rn
import argparse as _ar


def parse_args() -> _ar.Namespace:
    """
    Parse the command line arguments and perform some validation on the
    arguments.
    """
    description = 'split.py: spliting an EXYZ file.'
    parser = _ar.ArgumentParser(description=description)
    parser.add_argument('-i', '--input', required=True, type=str,
                        help='The input EXYZ file.')
    parser.add_argument('--ranges', required=True, type=str,
                        help='The frame ranges. For example, "0:2,4,7:9" ' +
                             'will select frames: [0,1,4,7,8]')
    parser.add_argument('--nframes', type=int,
                        help='Number of the frames to randomly select.')
    parser.add_argument('--perrange', action='store_true',
                        help='Select NFRAMES from each range.')
    if (len(_sys.argv) == 1):
        parser.print_help()
        exit()
    return parser.parse_args()


def read_frame(fp: _io.TextIOWrapper, n_frame: int) -> tuple:
    """
    Read one frame from the input stream.
    """
    lines = []
    n_atoms = fp.readline()
    if (n_atoms == ''):
        return (0, [])
    else:
        n_atoms = int(n_atoms)
    for i in range(0, (n_atoms + 1)):
        line = fp.readline()
        if (i == 0):
            if (not ('energy' in line or 'energies' in line)):
                message = 'Frame {:d} contains a bad comment line: "{}".'
                raise RuntimeError(message.format(n_frame, line.strip()))
        else:
            words = line.strip().split(' ')
            if ((len(words) - words.count('')) < 4):
                message = 'Frame {:d} contains a bad data line: "{}". ' + \
                          'This may be caused by the mismatching between' + \
                          'the given atom number and the data line number.'
                raise RuntimeError(message.format(n_frame, line.strip()))
        if (line != ''):
            lines.append(line.strip())
    if (len(lines) != (n_atoms + 1)):
        message = 'Mismatching between atom number {:d} '.format(n_atoms) + \
                  'and xyz data:\n'
        for line in lines:
            message += line + '\n'
        raise RuntimeError(message)
    return (n_atoms, lines)


def parse_input_frame_list(frame_list: str) -> list:
    """
    Parse the atom selections.
    """
    result = []
    for s in frame_list.split(','):
        if (':' not in s):
            if (not s.isnumeric()):
                raise SyntaxError('Unknown frame index: ' + s)
            result.append(int(s))
        else:
            l = s.split(':')
            if (not l[0].isnumeric() or len(l) != 2 or not l[1].isnumeric()):
                raise SyntaxError('Unknown frame range: ' + s)
            if (int(l[0]) >= int(l[1])):
                message = 'Bad frame range [{:d}, {:d}]! '
                message = message.format(int(l[0]), int(l[1]))
                message += 'The first index should be smaller than the ' + \
                           'second index!'
                raise RuntimeError(message)
            result.append(list(range(int(l[0]), int(l[1]))))
    return result


def make_frame_list(frame_ranges: list,
                    n_frames: int,
                    per_range: bool) -> list:
    """
    Get indices of the frames to write.
    """
    frame_list = []
    for element in frame_ranges:
        if (type(element) == int):
            frame_list.append(element)
        elif (type(element) == list):
            if (per_range and n_frames is not None):
                if (n_frames < len(element)):
                    element = _rn.sample(element, n_frames)
            for i in element:
                frame_list.append(i)
    if ((n_frames is not None) and not per_range):
        frame_list = _rn.sample(frame_list, n_frames)
    frame_list = list(set(frame_list))
    _rn.shuffle(frame_list)
    return frame_list


def print_frame(n_atoms: int, lines: list):
    """
    Print one frmae to the screen.
    """
    print(n_atoms)
    for line in lines:
        print(line)


def loop(file_name: str, frame_list: list):
    """
    Loop through the input file.
    """
    count = 0
    lines = ['null']
    frames = []
    fp = open(file_name, 'r')
    while True:
        n_atoms, lines = read_frame(fp, count)
        frames.append((n_atoms, lines))
        if (len(lines) == 0):
            break
        count += 1
    fp.close()
    n_frames_total = len(frames)
    for i in frame_list:
        if (i >= n_frames_total):
            message = 'Can not read frames: {:d}'.format(i)
            raise RuntimeError(message)
        else:
            print_frame(frames[i][0], frames[i][1])
    del frames


if (__name__ == '__main__'):
    parser = parse_args()
    if (parser.nframes is not None and parser.nframes <= 0):
        raise RuntimeError('The NFRAMES paramter should be positive!')
    frame_list = parse_input_frame_list(parser.ranges)
    frame_list = make_frame_list(frame_list, parser.nframes, parser.perrange)
    print("Selected frames:", file=_sys.stderr)
    for i in frame_list:
        print(i, file=_sys.stderr)
    loop(parser.input, frame_list)
