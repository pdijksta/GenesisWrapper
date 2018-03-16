
class GenesisParserError(KeyError):
    pass

class GenesisInputParser(dict):
    comment_chars = ('!', '%', '#',)

    def __init__(self, file_, debug=False):
        self.file_ = file_

        with open(self.file_, 'r') as f:
            lines = f.readlines()

        inside, outside = 0, 1
        status = outside
        blockname, blocklines = None, None

        for line in lines:
            line = line.strip()
            if debug:
                print(line)
            if any(line.startswith(c) for c in self.comment_chars):
                pass
            elif status == outside:
                if line.startswith('&'):
                    status = inside
                    blockname = line[1:]
                    blocklines = []
            elif status == inside:
                if line == '&end':
                    status = outside
                    self.addblock(blockname, blocklines)
                else:
                    blocklines.append(line)


    def addblock(self, blockname, blocklines):
        dd = {}
        for line in blocklines:
            try:
                key, value = line.split('=')
            except:
                print(line)
                raise
            key = key.strip()
            value = value.strip()

            try:
                value = float(value)
            except ValueError:
                pass

            if key in dd:
                print(blocklines)
                raise GenesisParserError('Key %s specified more than once!' % key)
            else:
                dd[key] = value

        if 'label' in dd:
            key = dd['label']
        else:
            key = blockname

        if key in self:
            raise GenesisParserError('Block %s occurs more than once!' % key)

        self[key] = dd

