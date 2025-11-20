// santorini board size

pub const BOARD_SIZE: usize = 5;

//tower level 3 
pub const MAX_LEVEL:u8 =3;

//position on board(X,Y)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]

pub struct Pos{ 
    pub x: u8,
    pub y: u8,
}

#[derive(Clone, Copy,Debug)]
pub struct Action{
    //index of worker
    pub worker_index: usize,
    //tagrt pos 
    pub move_to: Pos,
    //where to build 
    pub build_at: Pos,
}
impl Pos{
    // create a new pos, no bound check yet
    pub fn new(x:u8, y:u8) -> Self{
        Self {x,y}
    }

    //check pos is on 5x5 board
    pub fn in_bounds(&self) -> bool {
        (self.x as usize) < BOARD_SIZE && (self.y as usize) < BOARD_SIZE
    }
    //return max 8 neighbor position 
    pub fn neighbors(&self) -> Vec<Pos> {
        let directions: &[(i8,i8)] = &[
            (-1,-1),(0,1),(1,-1),
            (-1,0),       (1,0),
            (-1,1),(0,1), (1,1),
        ];

        let mut result = Vec::new();

        for (dx,dy) in directions{
            //convert cuurent xy to i16 (can add negative num safely)
            let nx =self.x as i16 +*dx as i16;
            let ny =self.y as i16 +*dy as i16;

            // check if still inside 0, BOARD_SIZE
            if nx >= 0 && ny >= 0 && nx <BOARD_SIZE as i16 && ny< BOARD_SIZE as i16{
                result.push(Pos {
                    x:nx as u8,
                    y:ny as u8
                });
            }
        }
         result 
    }
}

// cell tower
#[derive(Clone, Copy, Debug)]

pub struct Cell {
    // height of tower 0 - 4 when 4 is dome
    pub height:u8,
}

impl Cell{
    pub fn new() -> Self{
        Self{ height : 0}
    }
}

// worker on the board
#[derive(Clone, Copy, Debug)]
pub struct Worker{
    // which player own this worker (0,1,2,...)
    pub owner: u8,
    // position on board or none 
    pub pos: Option<Pos>,
}

#[derive(Clone, Debug)]
pub struct Board{
    // 2D array cell
    pub cells:[[Cell;BOARD_SIZE]; BOARD_SIZE],
    // workers in the game  2-3 player 2 worker each so 6 max
    pub workers:Vec<Worker>,
    //index of player whose turn it is 
    pub current_player:u8,
    //total num of player 2 or 3
    pub num_players:u8,
}

impl Board{
    // create an empty board
    pub fn new(num_players:u8) -> Self{
        //create 5x5 array of cell::new()
        let row = [Cell::new(); BOARD_SIZE];
        let cells = [row; BOARD_SIZE];

        Self{
            cells,
            workers:Vec::new(),
            current_player:0,
            num_players,

        }
    }

    // get a reference to a cell at given position
    pub fn cell(&self, pos:Pos) -> &Cell {
        &self.cells[pos.y as usize][pos.x as usize]
    }

    // get a mutabel reference to cell so it height be changable

    pub fn cell_mut(&mut self, pos: Pos) -> &mut Cell{
        &mut self.cells[pos.y as usize][pos.x as usize]
    }

    //add worker at given position
    pub fn add_worker(&mut self, owner:u8, pos:Pos){
        self.workers.push(Worker{
            owner, pos:Some(pos),
        });
    }

    // return true if any workers stand on this pos
    pub fn is_occupied(&self,pos: Pos) -> bool{
        self.workers.iter().any(|w| w.pos == Some(pos))
    }
    // return worker index that stand there
    pub fn worker_index_at(&self, pos: Pos) -> Option<usize> {
        self.workers
            .iter()
            .position(|w| w.pos == Some(pos))
    }
    // return immutable ref to worker at this pos
    pub fn worker_at(&self, pos:Pos) -> Option<&Worker> {
        self.worker_index_at(pos).map(|idx| &self.workers[idx])
    }
    // return mutable ref to worker at this pos
    pub fn worker_at_mut(&mut self, pos:Pos) -> Option<&mut Worker>{
        if let Some(idx) = self.worker_index_at(pos){
            Some(&mut self.workers[idx])
        } else{
            None
        }
    }
    // get height
    pub fn height_at(&self, pos:Pos) -> u8{
        self.cell(pos).height
    }

    // Santorini rules 
    // destination must be a neighbor
    // destination must not be occupied
    // you can climb at most +1 level
    // you cannot move onto a dome (height >= 4)
    pub fn can_move_worker_to(&self, worker_index:usize, to:Pos) ->bool {
        //1 get worker
        let worker = match self.workers.get(worker_index){
            Some(w)=> w,
            None => return false,
        };
        // 2 worker must be on board
        let from = match worker.pos{
            Some(p)=> p,
            None=> return false,
        };
        // 3 target must be in bounds
        if !to.in_bounds(){
            return false;
        }
        // 4 target must be neighbor
        let is_neighbor = from
            .neighbors()
            .into_iter()
            .any(|n|n == to);

        if !is_neighbor{
            return false;
        }
        // 5 cant move on another worker
        if self.is_occupied(to){
            return false;
        }
        //6 can climb max +1
        let from_h =self.height_at(from);
        let to_h = self.height_at(to);
        // cant move on lvl 4 dome
        if to_h>=4{
            return false;
        }
        //cant climb more than h+1
        if to_h > from_h +1{
            return false;
        }
        
        true

        

    }
    // return pos that are allowed to move
    pub fn legal_moves_for_worker(&self, worker_index:usize) -> Vec<Pos> {
        let worker =match self.workers.get(worker_index){
            Some(w)=> w,
            None => return Vec::new(),
        };

        let from = match worker.pos{
            Some(p) => p,
            None => return Vec::new(),
        };

        let mut moves = Vec::new();

        for n in from.neighbors() {
            if self.can_move_worker_to(worker_index, n) {
                moves.push(n);
            }
        }

        moves
    }
    //build rule
    // - `at` must be on the board
    // - `at` must be a neighbor of `from`
    // - `at` must not be the same as `from`
    // - `at` must not be occupied by a worker
    // - height at `at` must be <= 3 (you can't build on a dome; max height becomes 4)
    pub fn can_build_from(&self,from:Pos, at:Pos) -> bool {
        if !at.in_bounds(){
            return false;
        }
        //dont build on ur own cell
        if at == from{
            return false;
        }
        // must be neighbor of from
        let is_neighbor = from
            .neighbors()
            .into_iter()
            .any(|n| n == at);

        if !is_neighbor {
            return false;
        }

        // cant build on worker
        if self.is_occupied(at){
            return false;
        }
        // not build above dome
        let h = self.height_at(at);
        if h>=4 {
            return false;
        }

        true
    }

    //legal build pos  'from is the current worker pos and  at is the build cell'
    pub fn legal_builds_from(&self, from:Pos) -> Vec<Pos> {
        let mut builds = Vec::new();

        for at in from.neighbors(){
            if self.can_build_from(from, at){
                builds.push(at);
            }
        }
        builds
    }
    // Returns `true` if the worker was moved, `false` otherwise. and move if legal
    pub fn move_worker_to(&mut self, worker_index: usize, to:Pos) -> bool {
        if !self.can_move_worker_to(worker_index, to){
            return false;
        }

        if let Some(worker) =self.workers.get_mut(worker_index){
            worker.pos =Some(to);
            true
        } else{
            false
        }
    }
    
}
    