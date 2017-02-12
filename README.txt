Usage:

    let mut hmm = karma::Karma::new(20, 10);

    hmm.randomize();

    hmm.train(&vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9], None);
    hmm.train(&vec![0, 1, 4, 5, 7, 9, 0, 4, 7, 8, 2], Some(0.05));

    let out = hmm.evaluate(&vec![5, 6, 7, 8]);

    println!("{:?}", out);

┌────────────────────────────────────────────────────────────┐
│                   _  __   _   ___ __  __   _                     │
│                  | |/ /  /_\ | _ \  \/  | /_\                    │
│                  | ' <  / _ \|   / |\/| |/ _ \                   │
│                  |_|\_\/_/ \_\_|_\_|  |_/_/ \_\                  │
│                                                                  │
└────────────────────────────────────────────────────────────┘
                                          PsychonautWiki. REL/FOUI

~~ MIT License ~~

Copyright (c) 2017 Kenan Sulayman
Copyright (c) 2017 The PsychonautWiki Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
