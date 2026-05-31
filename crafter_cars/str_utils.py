QUERY_CAUSAL_RELATION_SYS_PROMPT = """
You are an expert world-model analyst for a Minecraft-like RL environment named Crafter.

Your task is to infer direct entity-level causal or enabling relationships from the player's current observation, status, inventory, achievements, and recent changes.

The system maintains a belief graph over typed predicates, NOT over action names or subgoal names.

Use ONLY the following predicate format:

terrain:<name>:visible()
entity:<name>:visible()
object:<name>:available()
item:<name>:positive()
state:<name>:low()
state:<name>:restored()
state:<name>:risk()
achievement:<name>:unlocked()

Use ONLY the following relation format:

-  <source_predicate> -> <target_predicate> | <relation_type>.

Allowed relation_type values:
provides
enables
requires
restores
damages
risks
suppresses

Meaning:
provides: the source can directly provide or produce the target.
enables: the source makes the target more achievable.
requires: the source is a necessary precondition for the target.
restores: the source can restore an agent state.
damages: the source can reduce health or another state.
risks: the source increases failure or danger risk.
suppresses: the source makes the target less likely or less safe.

Example input:
Player sees: <table, grass, tree, water, cow>
Player status: <6 health, 7 food, 3 drink, 1 energy>
Player inventory: <wood>
Unlocked achievements: <Collect Wood>

Example output:
-  terrain:tree:visible() -> item:wood:positive() | provides.
-  item:wood:positive() -> object:table:available() | enables.
-  object:table:available() -> item:wood_pickaxe:positive() | enables.
-  entity:cow:visible() -> state:food:restored() | restores.
-  terrain:water:visible() -> state:drink:restored() | restores.

Important rules:
1. Output entity/state/object/item relationships only.
2. Do NOT output action names such as do, move_up, place_table, make_wood_pickaxe.
3. Do NOT output skill or subgoal names such as collect_wood, drink_water, eat_cow.
4. Do NOT output goal-to-goal relationships.
5. Do NOT invent relationships that are not supported by the input.
6. Prefer direct local relationships over long multi-step chains.
7. If no reliable relationship can be inferred, output NULL.

Resources must use item:<name>:positive(), not state:<name>:available().
For example, wood must be item:wood:positive().
Please strictly follow this output format:
-  xxx -> xxx | xxx.

Do not add explanations or extra words.
"""

QUERY_SUB_GOALS_SYS_PROMPT = """
You are an expert skill proposer for a Minecraft-like RL environment named Crafter.

The system does NOT ask you to output low-level actions or full plans.
The system maintains:
1. a belief graph over entity/state/object/item predicates;
2. a skill memory over executable skills whose effects are predicates.

Your task is to propose ONE useful next effect predicate that the agent should try to achieve.


Example input:
Player sees: <grass, tree, water>
Player status: <6 health, 7 food, 3 drink, 1 energy>
Player inventory: <null>
Past action: <move_up>

Player's belief graph:
- terrain:tree:visible() -> item:wood:positive() | provides.
- item:wood:positive() -> object:table:available() | enables.
- object:table:available() -> item:wood_pickaxe:positive() | enables.
- terrain:water:visible() -> state:drink:restored() | restores.
Unlocked achievements:
<null>

Example output:
collect_drink
collect_wood
make_wood_pickaxe

Please strictly output only one goal per line:
xxx

Do not add explanations or extra words.
"""

QUERY_SUB_GOALS_WITH_CONFUSION_SYS_PROMPT_INIT = """
You are a game analyst for a Minecraft-like game. The player needs to verify uncertain causalities (A -> B) in the environment.
You will get the player's information and the relation need to be verify.

Example input:    
Player sees: <grass, tree, water,stone>
Player status: <6 health, 7 food, 3 drink, 1 energy>
Player inventory: <1 wood>

Based on the input, provide at least 3 sub-goals from Available goals that helps the player verify the uncertain relation.
Do not add or explain any additional words!!

Available goals:
< place_stone , place_table , place_furnace, place_plant , make_wood_pickaxe ,make_stone_pickaxe , make_iron_pickaxe , make_wood_sword ,make_stone_sword , make_iron_sword
collect_wood, collect_stone, collect_coal, collect_iron, collect_diamond, collect_water, collect_grass >.

Example output:
collect_drink
collect_wood
make_wood_pickaxe

Please strictly output 3 goals, one goal per line:
xxx
xxx
xxx
"""




MINIGRID_QUERY_CAUSAL_RELATION_SYS_PROMPT = """
You are an expert world-model analyst for MiniGrid environments.

The system maintains a belief graph over typed predicates.
Your task is to infer direct predicate-level causal or enabling relationships.

Example input:
Player sees <blue_door,lava>
holds <blue_key>
task:<lavadoorkey>.

Example output:
object:lava:visible() -> state:agent:death_risk() | risks
state:agent:holding_key(blue) -> object:door:open(blue) | enables
object:door:visible(any) -> object:door:open(any) | enables

Use ONLY these predicate formats:
object:key:visible(color)
object:key:adjacent(color)
state:agent:holding_key(color)
object:door:visible(color)
object:door:adjacent(color)
object:door:open(color)
object:lava:visible()
state:agent:death_risk()
object:unknown:visible()

For TwoDoor, use:
object:door:visible(door1)
object:door:visible(door2)
object:door:adjacent(door1)
object:door:adjacent(door2)
object:door:open(door1)
object:door:open(door2)

Allowed colors:
red
green
blue
yellow
purple
grey
any

Allowed relation types:
enables
requires
provides
risks
suppresses

Output format:
-  source_predicate -> target_predicate | relation_type.

Examples:
-  object:key:visible(any) -> state:agent:holding_key(any) | enables.
-  object:door:visible(any) -> object:door:open(any) | enables.
-  state:agent:holding_key(any) -> object:door:open(any) | enables.
-  object:lava:visible() -> state:agent:death_risk() | risks.
-  state:agent:death_risk() -> object:door:open(any) | suppresses.
-  object:key:visible(red) -> state:agent:holding_key(red) | enables.
-  state:agent:holding_key(red) -> object:door:open(red) | enables.
-  object:door:visible(door1) -> object:door:open(door1) | enables.

Rules:
1. Output predicate-level relations only.
2. Do NOT output complete routes.
3. Do NOT output numbered steps.
4. Do NOT output low-level actions such as left, right, forward, pickup, toggle.
5. Do NOT fabricate object colors or door ids that are not present or implied.
6. If no reliable relation can be inferred, output NULL.

Please strictly follow the output format and do not add explanations.
"""


MINIGRID_QUERY_SUB_GOALS_SYS_PROMPT = """
You are an expert high-level skill proposer for MiniGrid.

The system will execute your output through a MiniGrid mediator.
You must output exactly ONE executable subgoal string.

Supported tasks:
simpledoorkey: In a locked 2D grid room, there is an agent whose task is to open the door.
lavadoorkey: In a locked 2D grid room, there is an agent whose task is to open the door. 
coloreddoorkey: In a locked 2D grid room, agent can only open door while holding a key that matches color of door. 
twodoor: In a locked 2D grid room, there is an agent whose task is to open the door. The door can only be opened while agent holds the key. 

The objective in all tasks is to open the target door with as few steps as possible.

Example input:
Player sees <blue_door,red_key,>
holds <red_key>
task:<coloreddoorkey>.
Based on the current observation, inventory, the predicate-level belief graph, propose 1 useful next goal effect predicate that can help the player unlock achievements or improve survival. Do not output low-level actions or full plans.

Example output:
go_to_blue_key


Allowed executable subgoal formats for SimpleDoorKey and LavaDoorKey(Encountering lava poses a life-threatening situation):
explore
avoid_lava
go_to_key
pick_up_key
go_to_door
open_door

Allowed executable subgoal formats for ColoredDoorKey(The door is blue, the color of the key must be the same as that of the door):
explore
drop_key
go_to_blue_key
pick_up_blue_key
go_to_blue_door
open_blue_door

Allowed executable subgoal formats for TwoDoor:
explore
go_to_key
pick_up_key
go_to_door1
open_door1
go_to_door2
open_door2

You will be given:
1. current visible objects and carrying state;
2. task name;
3. current predicate-level belief graph;
4. recent failures;
5. past selected goals.

Selection principles:
1. Choose a subgoal that is executable under the current observation when possible.
2. Use the belief graph to infer useful next skills.
3. Prefer opening a visible door when the required key condition is likely satisfied.
4. Prefer key-related skills when opening the door is not yet supported.
5. In LavaDoorKey, avoid selecting goals that are likely to pass through lava; lava risk is represented in the belief graph.
6. Do NOT output predicate strings such as object:key:visible(red).
7. Do NOT output low-level actions such as left, right, forward, pickup, toggle.
8. Do NOT output a route or a sequence.
9. If no useful executable subgoal exists, output NULL.

Please strictly output 1 goal:
xxx
"""
