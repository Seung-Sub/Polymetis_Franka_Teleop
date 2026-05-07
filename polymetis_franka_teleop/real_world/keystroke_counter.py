from collections import defaultdict
from threading import Lock

# Lazy / X-tolerant imports so the module can be imported on a headless host
# (e.g. when running `--help` over SSH without DISPLAY). The Listener subclass
# will still raise at instantiation time if pynput cannot attach to X.
try:
    from pynput.keyboard import Key, KeyCode, Listener
except Exception:                  # pragma: no cover — headless fallback
    Listener = object               # type: ignore[assignment,misc]

    class _StubKey:
        space = backspace = enter = esc = None
    Key = _StubKey()                # type: ignore[assignment]

    class KeyCode:                  # type: ignore[no-redef]
        def __init__(self, char=None):
            self.char = char


class KeystrokeCounter(Listener):
    def __init__(self):
        self.key_count_map = defaultdict(lambda:0)
        self.key_press_list = list()
        self.lock = Lock()
        super().__init__(on_press=self.on_press, on_release=self.on_release)
    
    def on_press(self, key):
        with self.lock:
            self.key_count_map[key] += 1
            self.key_press_list.append(key)
    
    def on_release(self, key):
        pass
    
    def clear(self):
        with self.lock:
            self.key_count_map = defaultdict(lambda:0)
            self.key_press_list = list()
    
    def __getitem__(self, key):
        with self.lock:
            return self.key_count_map[key]
    
    def get_press_events(self):
        with self.lock:
            events = list(self.key_press_list)
            self.key_press_list = list()
            return events

if __name__ == '__main__':
    import time
    with KeystrokeCounter() as counter:
        try:
            while True:
                print('Space:', counter[Key.space])
                print('q:', counter[KeyCode(char='q')])
                time.sleep(1/60)
        except KeyboardInterrupt:
            events = counter.get_press_events()
            print(events)
