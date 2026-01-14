# Overview
**AI Jazz Solo Generator** is a React+Flask web application that lets you input a jazz style, tempo, and chord progression through the GUI, and **generates a MIDI solo** for you based on that progression.
It is perfect for those looking to add a jazz inspired solo on top of an existing chord progression. The model used for generation is a custom encoder-decoder Transformer trained on 400
solos from the **[Weimar Jazz Database](https://jazzomat.hfm-weimar.de/dbformat/dboverview.html)**. 

You can access the web application here: [AI Jazz Solo Generator](https://aijazz.diegojmejia.com/).

# Features / How To Use
- **Solo Settings GUI**:
  - **Filename:** Custom filename up to 100 characters
  - **Tempo:** (In BPM) tempo range from 60-300
  - **Style:** The style of jazz, with the options (Bebop, Postbop, Hardbop, Swing, and Cool)
  - **Key:** The base key for the solo to be in
- **Chord Progression GUI**:
  - **Add or remove custom bars (4/4 time only)**.
  - Bars can be **copied and pasted** onto other bars in order to speed up chord progression creation
  - Each bar has 4 beats, where each beat lets you:
    - **Change the key and the quality** such as (Db Major)
    - **Copy chords and paste them onto other beats** for faster usage
- **Generation:**
  - Once all the above settings are set, hit **Generate** in order to generate the solo
  - This process may take around 5-30 seconds depending on the internet connection and length of the chord progression.
  - Once done, the MIDI file will be downloaded onto your system and you may open it up using a DAW or a MIDI player.

# License
- This project is licensed under the MIT License
