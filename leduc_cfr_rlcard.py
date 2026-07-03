"""Step 2 of the research ladder: tabular CFR on Leduc Hold'em via RLCard.

Leduc Hold'em (6 cards, two rounds, one community card) is the standard
mid-size benchmark between Kuhn poker and real Texas Hold'em. RLCard's
CFRAgent is a tabular chance-sampling CFR implementation; this script trains
it and periodically evaluates average payoff per hand against a random agent.

Usage:
    .venv\\Scripts\\python leduc_cfr_rlcard.py --episodes 2000 --eval-every 500

Notes for the next rungs:
  * Payoff vs a random agent is a weak, *relative* metric -- fine as a smoke
    test, but the dissertation-grade metric is exploitability. OpenSpiel's
    `exploitability.nash_conv` computes this exactly for Leduc.
  * Swap CFRAgent for rlcard.agents.NFSPAgent / DQNAgent (requires torch) to
    reproduce the classic CFR-vs-deep-RL comparison on the same environment.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import rlcard
from rlcard.agents import CFRAgent, RandomAgent
from rlcard.utils import set_seed, tournament


class CompatCFRAgent(CFRAgent):
    """rlcard 1.2.0 calls ndarray.tostring(), removed in numpy 2.0.

    Overrides the two affected methods with tobytes() equivalents so the
    installed package stays untouched.
    """

    def get_state(self, player_id):
        state = self.env.get_state(player_id)
        return state["obs"].tobytes(), list(state["legal_actions"].keys())

    def eval_step(self, state):
        legal = list(state["legal_actions"].keys())
        probs = self.action_probs(state["obs"].tobytes(), legal, self.average_policy)
        action = np.random.choice(len(probs), p=probs)
        info = {
            "probs": {
                state["raw_legal_actions"][i]: float(probs[legal[i]])
                for i in range(len(legal))
            }
        }
        return action, info


def main():
    parser = argparse.ArgumentParser(description="Tabular CFR on Leduc Hold'em")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-games", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-dir", default="./leduc_cfr_model")
    args = parser.parse_args()

    set_seed(args.seed)

    # CFR needs step_back to traverse the tree; evaluation env plays forward only.
    env = rlcard.make("leduc-holdem", config={"seed": args.seed, "allow_step_back": True})
    eval_env = rlcard.make("leduc-holdem", config={"seed": args.seed})

    agent = CompatCFRAgent(env, model_path=os.path.join(args.model_dir, "cfr_model"))
    eval_env.set_agents([agent, RandomAgent(num_actions=eval_env.num_actions)])

    print(f"Training tabular CFR on Leduc Hold'em for {args.episodes} episodes")
    for episode in range(1, args.episodes + 1):
        agent.train()
        if episode % args.eval_every == 0 or episode == args.episodes:
            payoff = tournament(eval_env, args.eval_games)[0]
            print(f"episode {episode:>6}: avg payoff vs random = {payoff:+.3f} chips/hand "
                  f"({len(agent.policy)} infosets)")

    agent.save()
    print(f"Model saved to {args.model_dir}")


if __name__ == "__main__":
    main()
