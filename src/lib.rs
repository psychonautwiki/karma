extern crate rand;

use rand::Rng;

pub struct Karma {
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
    pub fn new(states: i64, items: i64) -> Karma {
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
        if new_size <= self._max_size {
            return;
        }

        self._max_size = new_size;
        self._tmp = Box::new(vec!(0f64; (self.N * (3 * new_size + self.N)) as usize));
    }

    pub fn randomize(&mut self) {
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

    pub fn evaluate(&mut self, outputs: Vec<i64>) -> f64 {
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

        while t < outputs.len() {
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

                sum = 0f64;

                for i in 0..states {
                    sum = sum + (self._tmp[(states + i) as usize] * self.next_probs[l as usize]);

                    l = l + states;
                }

                self._tmp[(states + j) as usize] = sum + self.out_probs[k as usize];

                k = k +1;
            }

            t = t + 1;
        }

        let mut sum = 0f64;

        for i in 0..states {
            let offset = match t & 1 {
                1 => i,
                _ => (states + i)
            };

            sum = sum + (self._tmp[offset as usize]);
        }

        sum
    }

    pub fn train(&mut self, outputs: Vec<i64>, rate: Option<f64>) {
        let rate = match rate {
            Some(rate) => rate,
            None => 0.05f64
        };

        let states = self.N;
        let items = self.O;

        let S = outputs.len() as i64;

        let outputs_N: Vec<i64> = outputs.clone()
                                         .into_iter()
                                         .map(|x| x * states)
                                         .collect();

        self._resize_tmp(S);

        let BETA: i64 = states * items;
        let GAMMA = 2 * BETA;
        let KAPPA = 3 * BETA;

        // Expectation - step 1 (computing alpha and beta)

        for j in 0..states {
            let j = j as usize;

            self._tmp[j] = self.init_probs[j] * self.out_probs[j + (outputs_N[0] as usize)];
        }

        for i in 1..S {
            let mut l = 0i64;

            println!("{:?}", self._tmp);

            for j in 0..states {
                let mut sum = 0f64;

                for k in 0..states {
                    self._tmp[(KAPPA + l) as usize] = 0f64;

                    sum = sum
                        + ((
                            self._tmp[((i - 1) * states + k) as usize]
                          * self.next_probs[(k * states + j) as usize]
                        ) as f64);

                    l = l + 1;
                }

                self._tmp[(i * states + j) as usize] =
                      sum
                    * self.out_probs[(j + outputs_N[i as usize]) as usize];
            }

            println!("{:?}", self._tmp);

            l = (S - 1) * states;

            for j in 0..states {
                self._tmp[(BETA + l) as usize] = 1f64;

                l = l + 1;
            }

            for i in (0..S-1).rev() {
                l = i * states;

                for j in 0..states {
                    let index = (BETA + l) as usize;

                    self._tmp[index] = 0f64;

                    for k in 0..states {
                        self._tmp[index] =
                              self._tmp[index]
                            + (
                                  self.next_probs[(j * states + k) as usize]
                                * self.out_probs[(k + outputs_N[(i + 1) as usize]) as usize]
                                * self._tmp[(BETA + (i + 1) * states + k) as usize]
                            );
                    }

                    l = l + 1;
                }
            }

            // Expectation - step 2 (computing gamma and kappa)

            for i in i..S {
                let mut sum = 0f64;

                for k in 0..states {
                    sum = sum
                          + (
                               self._tmp[(i * states + k) as usize]
                             * self._tmp[(BETA + i * states + k) as usize]
                          );
                }

                for j in 0..states {
                    self._tmp[(GAMMA + i * states + j) as usize] =
                          self._tmp[(i * states + j) as usize]
                        * self._tmp[(BETA + i * states + j) as usize]
                        / sum;
                }

                if i == S - 1 {
                    break;
                }

                let mut sum = 0f64;

                for j in 0..states {
                    for k in 0..states {
                        sum = sum
                            + (
                                 self._tmp[(i * states + j) as usize]
                               * self.next_probs[(j * states + k) as usize]
                               * self.out_probs[(k + outputs_N[(i + 1) as usize]) as usize]
                               * self._tmp[(BETA + (i + 1) * states + k) as usize]
                            );
                    }
                }

                let mut l = 0i64;

                for j in 0..states {
                    for k in 0..states {
                        let index = (KAPPA + l) as usize;

                        self._tmp[index] = self._tmp[index]
                                         + (
                                              self._tmp[(i * states + j) as usize]
                                            * self.next_probs[l as usize]
                                            * self.out_probs[(k + outputs_N[(i + 1) as usize]) as usize]
                                            * self._tmp[(BETA + (i + 1) * states + k) as usize]
                                            / sum
                                          );

                        l = l + 1;
                    }
                }

                // Maximum likelihood

                let mut l = 0i64;
                let mut del = 0f64;

                for i in 0..states {
                    let mut sum = 0f64;

                    for k in 0..(S - 1) {
                        sum = sum + self._tmp[(GAMMA + k * states + i) as usize];
                    }

                    for j in 0..states {
                        let index = l as usize;

                        del = self._tmp[KAPPA as usize + index]
                            / sum
                            - self.next_probs[index];

                        self.next_probs[index] = self.next_probs[index] + (del * rate);

                        l = l + 1i64;
                    }

                    sum = sum + self._tmp[(GAMMA + (S - 1) * states + i) as usize];

                    for j in 0..items {
                        let mut x = 0f64;

                        for k in 0..S {
                            let output = outputs[k as usize];
                            if output == j {
                                x = x + self._tmp[(GAMMA + k * states + i) as usize];
                            }
                        }

                        del = x / sum - self.out_probs[(i + j * states) as usize];

                        let index = (i + j * states) as usize;

                        self.out_probs[index] = self.out_probs[index] + (del * rate);
                    }

                    del = self._tmp[(GAMMA + i) as usize] - self.init_probs[i as usize];

                    self.init_probs[i as usize] = self.init_probs[i as usize] + (del * rate);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
    }
}
