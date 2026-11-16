"""Terminal progress without extra dependencies."""
import shutil
import sys


class ProgressLine:
    def __init__(self, mode='auto', stream=None):
        if mode not in ('auto', 'line', 'plain'):
            raise ValueError('Progress mode must be auto, line, or plain')
        self.stream = sys.stdout if stream is None else stream
        self.inline = mode == 'line' or (mode == 'auto' and self.stream.isatty())
        self.width = 0

    def update(self, text):
        if self.inline:
            # Leave a column free to avoid terminal wrapping and scrolling.
            limit = max(1, shutil.get_terminal_size((120, 24)).columns - 1)
            text = text[:limit]
            self.stream.write('\r' + text + ' ' * max(0, min(self.width, limit) - len(text)))
            self.width = len(text)
        else:
            self.stream.write(text + '\n')
        self.stream.flush()

    def finish(self):
        if self.inline and self.width:
            self.stream.write('\n')
            self.stream.flush()
            self.width = 0
