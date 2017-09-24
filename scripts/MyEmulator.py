from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import Card, Deck
from pypokerengine.utils.game_state_utils import restore_game_state
from pypokerengine.api.game import setup_config, start_poker
from pypokerengine.api.emulator import Emulator, ActionChecker, RoundManager,MessageBuilder, Const, Event, DataEncoder


class MyEmulator(Emulator):
    def run_until_my_next_action(self, game_state, my_uuid, my_messages):
        try:
            _ = self.mailbox
        except:
            self.mailbox = None
    
        if self.mailbox is not None and len(self.mailbox) > 0:
            self.mailbox += my_messages
        else:
#             print('start round')
            self.mailbox = []
            round_state = DataEncoder.encode_round_state(game_state)
            seats = round_state['seats']
            self.start_stack = [s['stack'] for s in seats if s['uuid'] == my_uuid][0]
            
        while game_state["street"] != Const.Street.FINISHED:
            next_player_pos = game_state["next_player"]
            next_player_uuid = game_state["table"].seats.players[next_player_pos].uuid     
#             print(next_player_uuid)
            next_player_algorithm = self.fetch_player(next_player_uuid)
            msg = MessageBuilder.build_ask_message(next_player_pos, game_state)["message"]
            if next_player_uuid == my_uuid:
#                 print(msg["valid_actions"])
                return game_state, msg["valid_actions"], msg["hole_card"], msg["round_state"]
            
            action, amount = next_player_algorithm.declare_action(msg["valid_actions"], msg["hole_card"], msg["round_state"])
            
            game_state, messages = RoundManager.apply_action(game_state, action, amount)
            self.mailbox += messages
        events = [self.create_event(message[1]["message"]) for message in self.mailbox]
        events = [e for e in events if e]
        self.mailbox = []
        
        round_state = DataEncoder.encode_round_state(game_state)
        seats = round_state['seats']
        end_stack = [s['stack'] for s in seats if s['uuid'] == my_uuid][0]
#         print('finish, reward', end_stack - self.start_stack, end_stack)

        if self._is_last_round(game_state, self.game_rule):
#             print('last_round')
            events += self._generate_game_result_event(game_state)
            
        return game_state, end_stack - self.start_stack
    
    def apply_my_action(self, game_state, action, bet_amount=0):
        updated_state, messages = RoundManager.apply_action(game_state, action, bet_amount)
        return updated_state, messages

