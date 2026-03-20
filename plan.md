Wait, my files were reset!
Why? Oh, did I run `git restore`? No. Did another subagent run `reset_all`? No.
Ah, my previous changes *were* discarded somehow. Wait! "I have implemented the fixes across the codebase" - wait, no, the test suite output showed I had to resolve merge conflicts or something? No, `git reset` was never run by me, but `src/coreason_actuator/semantic_extractor.py` has definitely reverted! Let me redo the changes.
