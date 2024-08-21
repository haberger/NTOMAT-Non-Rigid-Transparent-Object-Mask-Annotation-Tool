conda env create -f env.yml



i think architecture wise it would be better instead f keeping track about the activeness of a scene/image/object to use the menu elements to define what is active -> seems way cleaner.
So instead of changing stuff when a dropdown is selected we just read what stands inside the menues and filter them
Efficient passing is done vie a single state that consists of all the menu item values. We dont care about the rest???