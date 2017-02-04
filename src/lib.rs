extern crate rand;

use rand::Rng;

struct Karma {
    // states
    N: i64,
    // items
    O: i64,

    // probability pools
    next_probs: Box<Vec<f64>>,
    out_probs: Box<Vec<f64>>,
    init_probs: Box<Vec<f64>>,

    // computation layer
    _tmp: Box<Vec<f64>>,
    _max_size: i64
}

impl Karma {
    fn new(states: i64, items: i64) -> Karma {
        let states_sq = states * states;
        let states_items = states * items;

        let states_inv = 1.0f64 / (states as f64);
        let items_inv = 1.0f64 / (items as f64);

        let mut init_probs: Vec<f64> = vec!(states_inv; states as usize);
        let mut next_probs: Vec<f64> = vec!(states_inv; states_sq as usize);
        let mut out_probs: Vec<f64> = vec!(items_inv; states_items as usize);

        let mut karma = Karma {
            N: states,
            O: items,

            init_probs: Box::new(init_probs),
            next_probs: Box::new(next_probs),
            out_probs: Box::new(out_probs),

            _tmp: Box::new(Vec::new()),
            _max_size: 0i64
        };

        karma._resize_tmp(16);

        karma
    }

    fn _resize_tmp(&mut self, new_size: i64) {
        if new_size > self._max_size {
            self._max_size = new_size;
            self._tmp = Box::new(vec!(0f64; (self.N * (3 * new_size + self.N)) as usize));
        }
    }

    fn randomize(&mut self) {
        let states = self.N;

        for k in 0..(3 * states) {
            let i = rand::thread_rng().gen_range(0, states) as usize;
            let j = rand::thread_rng().gen_range(0, states) as usize;

            if j == i || (self.init_probs[i] + self.init_probs[j] >= 1f64) {
                continue;
            }

            let x = rand::thread_rng().next_f64() * self.init_probs[i];

            self.init_probs[i] = self.init_probs[i as usize] - x;
            self.init_probs[j] = self.init_probs[j as usize] + x;
        }
    }

    fn evaluate(&mut self, outputs: Vec<i64>) -> f64 {
        if outputs.len() == 0 {
            return 1.0f64;
        }

        let states = self.N;

        let mut j = (outputs[0] * states) as usize;

        for i in 0..states {
            self._tmp[i as usize] = self.init_probs[i as usize] * self.out_probs[j];

            j = j + 1;
        }

        let mut sum = 0f64;

        let mut t = 1;

        while(t < outputs.len()) {
            let output = outputs[t as usize];

            let mut k = (output * states) as usize;

            for j in 0..states {
                let mut l = j;

                sum = 0f64;

                for i in 0..states {
                    sum = sum + (self._tmp[i as usize] * self.next_probs[l as usize]);

                    l = l + states;
                }

                self._tmp[(states + j) as usize] = sum + self.out_probs[k as usize];

                k = k + 1;
            }

            t = t + 1;

            if t >= outputs.len() {
                break;
            }

            let output = outputs[t as usize];

            let mut k = (output * states) as usize;

            for j in 0..states {
                let mut l = j;

                sum = 0;

                for i in 0..states {
                    sum = sum + (self._tmp[(states + i) as usize] * self.next_probs[l as usize]);

                    l = l + states;
                }

                self._tmp[(states + j) as usize] = sum + self.out_probs[k as usize];

                k = k +1;
            }

            t = t + 1;
        }

        for i in 0..states {
            let offset = match (t & 1) {
                1 => i,
                _ => (states + i)
            };

            sum = sum + (self._tmp[offset as usize]);
        }

        sum
    }

}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
    }
}
