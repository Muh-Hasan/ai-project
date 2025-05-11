import argparse
import torch
from src.tetris import Tetris
import torch.serialization

try:
    from src.deep_q_network import DeepQNetwork
    torch.serialization.add_safe_globals(["src.deep_q_network.DeepQNetwork"])
except (ImportError, AttributeError):
    pass  


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Tetris with Extended Shapes""")

    parser.add_argument("--width", type=int, default=10, help="The common width for all images")
    parser.add_argument("--height", type=int, default=20, help="The common height for all images")
    parser.add_argument("--block_size", type=int, default=30, help="Size of a block")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--model", type=str, default="tetris",
                      help="Model filename to use (e.g., tetris_4500)")

    args = parser.parse_args()
    return args


def test(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    
    model_path = "{}/{}".format(opt.saved_path, opt.model)
    print(f"Loading model from {model_path}")
    
    try:
        if torch.cuda.is_available():
            model = torch.load(model_path, weights_only=False)
        else:
            model = torch.load(model_path, map_location=lambda storage, loc: storage, weights_only=False)
        
        print("Model loaded successfully")
    except TypeError:
        print("Trying with older PyTorch load method...")
        if torch.cuda.is_available():
            model = torch.load(model_path)
        else:
            model = torch.load(model_path, map_location=lambda storage, loc: storage)
        print("Model loaded successfully with older PyTorch method")
    
    model.eval()
    env = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size,
                use_extended_shapes=True)
    env.reset()
    if torch.cuda.is_available():
        model.cuda()
    
    try:
        while True:
            next_steps = env.get_next_states()
            next_actions, next_states = zip(*next_steps.items())
            next_states = torch.stack(next_states)
            if torch.cuda.is_available():
                next_states = next_states.cuda()
            predictions = model(next_states)[:, 0]
            index = torch.argmax(predictions).item()
            action = next_actions[index]
            _, done = env.step(action, render=True)

            if done:
                break
    except Exception as e:
        print(f"Error during gameplay: {e}")


if __name__ == "__main__":
    opt = get_args()
    test(opt)
