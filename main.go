package main

import (
	"context"
	"flag"
	"fmt"
	"math"
	"os"
	"os/signal"
	"runtime"
	"sort"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
	"unsafe"
)

type Move uint8

const (
	Rock Move = iota
	Paper
	Scissors
	Lizard
	Spock
	MaxMoves = 5
)

type GameResult uint8

const (
	Draw GameResult = iota
	Player1Win
	Player2Win
)

type GameState struct {
	WinMatrix    [MaxMoves][MaxMoves]GameResult
	MoveNames    [MaxMoves]string
	MoveSymbols  [MaxMoves]string
	ResultNames  [3]string
	BeatsMatrix  [MaxMoves]uint8
	CounterMoves [MaxMoves][MaxMoves]Move
}

func NewGameState() *GameState {
	gs := &GameState{
		MoveNames:   [MaxMoves]string{"Rock", "Paper", "Scissors", "Lizard", "Spock"},
		MoveSymbols: [MaxMoves]string{"🪨", "📄", "✂️", "🦎", "🖖"},
		ResultNames: [3]string{"Draw", "Player1 Win", "Player2 Win"},
	}

	gs.BeatsMatrix[Rock] = (1 << Scissors) | (1 << Lizard)
	gs.BeatsMatrix[Paper] = (1 << Rock) | (1 << Spock)
	gs.BeatsMatrix[Scissors] = (1 << Paper) | (1 << Lizard)
	gs.BeatsMatrix[Lizard] = (1 << Spock) | (1 << Paper)
	gs.BeatsMatrix[Spock] = (1 << Scissors) | (1 << Rock)

	gs.initializeMatrices()
	return gs
}

func (gs *GameState) initializeMatrices() {
	for m1 := Move(0); m1 < MaxMoves; m1++ {
		for m2 := Move(0); m2 < MaxMoves; m2++ {
			if m1 == m2 {
				gs.WinMatrix[m1][m2] = Draw
			} else if (gs.BeatsMatrix[m1] & (1 << m2)) != 0 {
				gs.WinMatrix[m1][m2] = Player1Win
			} else {
				gs.WinMatrix[m1][m2] = Player2Win
			}
		}
	}

	for opponentMove := Move(0); opponentMove < MaxMoves; opponentMove++ {
		var bestCounter Move
		for myMove := Move(0); myMove < MaxMoves; myMove++ {
			if gs.WinMatrix[myMove][opponentMove] == Player1Win {
				bestCounter = myMove
				break
			}
		}
		for myMove := Move(0); myMove < MaxMoves; myMove++ {
			gs.CounterMoves[opponentMove][myMove] = bestCounter
		}
	}
}

func (gs *GameState) EvaluateMove(move1, move2 Move) GameResult {
	return gs.WinMatrix[move1][move2]
}

func (gs *GameState) GetPayoff(result GameResult) float64 {
	switch result {
	case Player1Win:
		return 1.0
	case Draw:
		return 0.0
	case Player2Win:
		return -1.0
	}
	return 0.0
}

type MoveHistory struct {
	moves    []Move
	capacity int
	index    int
	count    int
}

func NewMoveHistory(capacity int) *MoveHistory {
	return &MoveHistory{
		moves:    make([]Move, capacity),
		capacity: capacity,
	}
}

func (h *MoveHistory) Add(move Move) {
	h.moves[h.index] = move
	h.index = (h.index + 1) % h.capacity
	if h.count < h.capacity {
		h.count++
	}
}

func (h *MoveHistory) GetSequence(length int) []Move {
	if length <= 0 || length > h.count {
		return nil
	}

	sequence := make([]Move, length)
	start := (h.index - length + h.capacity) % h.capacity

	for i := 0; i < length; i++ {
		sequence[i] = h.moves[(start+i)%h.capacity]
	}

	return sequence
}

func (h *MoveHistory) GetFrequencies() [MaxMoves]float64 {
	var frequencies [MaxMoves]float64
	if h.count == 0 {
		for i := range frequencies {
			frequencies[i] = 1.0 / MaxMoves
		}
		return frequencies
	}

	counts := [MaxMoves]int{}
	for i := 0; i < h.count; i++ {
		counts[h.moves[i]]++
	}

	for i := range frequencies {
		frequencies[i] = float64(counts[i]) / float64(h.count)
	}
	return frequencies
}

func (h *MoveHistory) Size() int {
	return h.count
}

type PatternMatcher struct {
	patterns map[string][MaxMoves]int
	orders   []int
}

func NewPatternMatcher() *PatternMatcher {
	return &PatternMatcher{
		patterns: make(map[string][MaxMoves]int),
		orders:   []int{1, 2, 3, 4},
	}
}

func (pm *PatternMatcher) Update(history *MoveHistory) {
	for _, order := range pm.orders {
		if history.Size() <= order {
			continue
		}

		sequence := history.GetSequence(order + 1)
		if len(sequence) != order+1 {
			continue
		}

		pattern := pm.encodePattern(sequence[:order])
		nextMove := sequence[order]

		counts := pm.patterns[pattern]
		counts[nextMove]++
		pm.patterns[pattern] = counts
	}
}

func (pm *PatternMatcher) PredictProbabilities(history *MoveHistory) [MaxMoves]float64 {
	var result [MaxMoves]float64
	totalWeight := 0.0

	for _, order := range pm.orders {
		if history.Size() < order {
			continue
		}

		sequence := history.GetSequence(order)
		if len(sequence) != order {
			continue
		}

		pattern := pm.encodePattern(sequence)
		counts, exists := pm.patterns[pattern]
		if !exists {
			continue
		}

		total := 0
		for _, count := range counts {
			total += count
		}

		if total == 0 {
			continue
		}

		weight := float64(total) * float64(order)
		totalWeight += weight

		for i := range result {
			probability := float64(counts[i]) / float64(total)
			result[i] += probability * weight
		}
	}

	if totalWeight == 0 {
		for i := range result {
			result[i] = 1.0 / MaxMoves
		}
		return result
	}

	for i := range result {
		result[i] /= totalWeight
	}

	return result
}

func (pm *PatternMatcher) encodePattern(moves []Move) string {
	if len(moves) == 0 {
		return ""
	}
	result := make([]byte, len(moves))
	for i, move := range moves {
		result[i] = byte(move) + '0'
	}
	return string(result)
}

type FastRNG struct {
	state uint64
}

func NewFastRNG(seed uint64) *FastRNG {
	if seed == 0 {
		seed = uint64(time.Now().UnixNano())
	}
	return &FastRNG{state: seed}
}

func (r *FastRNG) Next() uint32 {
	r.state ^= r.state << 13
	r.state ^= r.state >> 17
	r.state ^= r.state << 5
	return uint32(r.state)
}

func (r *FastRNG) NextFloat() float64 {
	return float64(r.Next()) / float64(^uint32(0))
}

type AIPlayer interface {
	Name() string
	MakeMove(ctx context.Context, opponentHistory *MoveHistory, gameState *GameState) Move
	UpdateStrategy(myMove, opponentMove Move)
	Reset()
}

type MinimaxNode struct {
	myMove       Move
	opponentMove Move
	depth        int
	isMaximizing bool
}

type AlphaBetaAI struct {
	name            string
	opponentHistory *MoveHistory
	myHistory       *MoveHistory
	patternMatcher  *PatternMatcher
	rng             *FastRNG
	maxDepth        int
	gameState       *GameState
}

func NewAlphaBetaAI(name string, maxDepth int) *AlphaBetaAI {
	seed := uint64(time.Now().UnixNano()) ^ uint64(uintptr(unsafe.Pointer(&name)))
	return &AlphaBetaAI{
		name:            name,
		opponentHistory: NewMoveHistory(512),
		myHistory:       NewMoveHistory(512),
		patternMatcher:  NewPatternMatcher(),
		rng:             NewFastRNG(seed),
		maxDepth:        maxDepth,
	}
}

func (ai *AlphaBetaAI) Name() string {
	return ai.name
}

func (ai *AlphaBetaAI) Reset() {
	ai.opponentHistory = NewMoveHistory(512)
	ai.myHistory = NewMoveHistory(512)
	ai.patternMatcher = NewPatternMatcher()
}

func (ai *AlphaBetaAI) UpdateStrategy(myMove, opponentMove Move) {
	ai.myHistory.Add(myMove)
	ai.opponentHistory.Add(opponentMove)
	ai.patternMatcher.Update(ai.opponentHistory)
}

func (ai *AlphaBetaAI) MakeMove(ctx context.Context, opponentHistory *MoveHistory, gameState *GameState) Move {
	ai.gameState = gameState

	if ai.opponentHistory.Size() < 2 {
		return Move(ai.rng.Next() % MaxMoves)
	}

	_, bestMove := ai.minimax(0, math.Inf(-1), math.Inf(1), true)
	return bestMove
}

func (ai *AlphaBetaAI) minimax(depth int, alpha, beta float64, maximizing bool) (float64, Move) {
	if depth >= ai.maxDepth {
		return ai.evaluateLeafNode(), Move(0)
	}

	opponentProbs := ai.patternMatcher.PredictProbabilities(ai.opponentHistory)

	if maximizing {
		maxEval := math.Inf(-1)
		var bestMove Move

		for myMove := Move(0); myMove < MaxMoves; myMove++ {
			var expectedValue float64

			for opponentMove := Move(0); opponentMove < MaxMoves; opponentMove++ {
				prob := opponentProbs[opponentMove]
				if prob < 0.001 {
					continue
				}

				gameResult := ai.gameState.EvaluateMove(myMove, opponentMove)
				payoff := ai.gameState.GetPayoff(gameResult)

				if depth < ai.maxDepth-1 {
					futureVal, _ := ai.minimax(depth+1, alpha, beta, false)
					payoff = 0.7*payoff + 0.3*futureVal
				}

				expectedValue += prob * payoff
			}

			if expectedValue > maxEval {
				maxEval = expectedValue
				bestMove = myMove
			}

			alpha = math.Max(alpha, expectedValue)
			if beta <= alpha {
				break
			}
		}

		return maxEval, bestMove
	} else {
		minEval := math.Inf(1)
		var bestMove Move

		for opponentMove := Move(0); opponentMove < MaxMoves; opponentMove++ {
			prob := opponentProbs[opponentMove]
			if prob < 0.001 {
				continue
			}

			var worstCaseValue float64 = math.Inf(1)

			for myMove := Move(0); myMove < MaxMoves; myMove++ {
				gameResult := ai.gameState.EvaluateMove(myMove, opponentMove)
				payoff := -ai.gameState.GetPayoff(gameResult)

				if payoff < worstCaseValue {
					worstCaseValue = payoff
				}
			}

			weightedValue := prob * worstCaseValue
			if weightedValue < minEval {
				minEval = weightedValue
				bestMove = opponentMove
			}

			beta = math.Min(beta, weightedValue)
			if beta <= alpha {
				break
			}
		}

		return minEval, bestMove
	}
}

func (ai *AlphaBetaAI) evaluateLeafNode() float64 {
	if ai.opponentHistory.Size() < 3 {
		return 0.0
	}

	recentMoves := ai.opponentHistory.GetSequence(min(10, ai.opponentHistory.Size()))
	if len(recentMoves) == 0 {
		return 0.0
	}

	frequencies := [MaxMoves]int{}
	for _, move := range recentMoves {
		frequencies[move]++
	}

	var entropy float64
	total := len(recentMoves)
	for _, freq := range frequencies {
		if freq > 0 {
			p := float64(freq) / float64(total)
			entropy -= p * math.Log2(p)
		}
	}

	maxEntropy := math.Log2(MaxMoves)
	return (maxEntropy - entropy) / maxEntropy
}

type MarkovState struct {
	transitions [MaxMoves][MaxMoves]int
	totals      [MaxMoves]int
	smoothing   float64
}

func NewMarkovState() *MarkovState {
	return &MarkovState{smoothing: 0.1}
}

func (ms *MarkovState) Update(prevMove, currentMove Move) {
	ms.transitions[prevMove][currentMove]++
	ms.totals[prevMove]++
}

func (ms *MarkovState) GetProbabilities(prevMove Move) [MaxMoves]float64 {
	var probs [MaxMoves]float64
	total := ms.totals[prevMove]

	if total == 0 {
		for i := range probs {
			probs[i] = 1.0 / MaxMoves
		}
		return probs
	}

	smoothedTotal := float64(total) + ms.smoothing*MaxMoves
	for i := Move(0); i < MaxMoves; i++ {
		smoothedCount := float64(ms.transitions[prevMove][i]) + ms.smoothing
		probs[i] = smoothedCount / smoothedTotal
	}

	return probs
}

type MarkovChainAI struct {
	name            string
	markovStates    map[int]*MarkovState
	opponentHistory *MoveHistory
	rng             *FastRNG
	maxOrder        int
	adaptiveWeights []float64
}

func NewMarkovChainAI(name string, maxOrder int) *MarkovChainAI {
	seed := uint64(time.Now().UnixNano()) ^ uint64(uintptr(unsafe.Pointer(&name)))
	weights := make([]float64, maxOrder)
	for i := range weights {
		weights[i] = float64(i+1) * float64(i+1)
	}

	return &MarkovChainAI{
		name:            name,
		markovStates:    make(map[int]*MarkovState),
		opponentHistory: NewMoveHistory(1024),
		rng:             NewFastRNG(seed),
		maxOrder:        maxOrder,
		adaptiveWeights: weights,
	}
}

func (ai *MarkovChainAI) Name() string {
	return ai.name
}

func (ai *MarkovChainAI) Reset() {
	ai.markovStates = make(map[int]*MarkovState)
	ai.opponentHistory = NewMoveHistory(1024)
}

func (ai *MarkovChainAI) UpdateStrategy(myMove, opponentMove Move) {
	ai.opponentHistory.Add(opponentMove)

	for order := 1; order <= ai.maxOrder && order < ai.opponentHistory.Size(); order++ {
		if ai.markovStates[order] == nil {
			ai.markovStates[order] = NewMarkovState()
		}

		sequence := ai.opponentHistory.GetSequence(order + 1)
		if len(sequence) == order+1 {
			prevMove := sequence[order-1]
			currentMove := sequence[order]
			ai.markovStates[order].Update(prevMove, currentMove)
		}
	}
}

func (ai *MarkovChainAI) MakeMove(ctx context.Context, opponentHistory *MoveHistory, gameState *GameState) Move {
	if ai.opponentHistory.Size() < 2 {
		return Move(ai.rng.Next() % MaxMoves)
	}

	prediction := ai.getBestPrediction()
	return ai.selectOptimalResponse(prediction, gameState)
}

func (ai *MarkovChainAI) getBestPrediction() [MaxMoves]float64 {
	var weightedPrediction [MaxMoves]float64
	totalWeight := 0.0

	for order := 1; order <= ai.maxOrder && order < ai.opponentHistory.Size(); order++ {
		state, exists := ai.markovStates[order]
		if !exists {
			continue
		}

		sequence := ai.opponentHistory.GetSequence(order)
		if len(sequence) != order {
			continue
		}

		lastMove := sequence[order-1]
		probs := state.GetProbabilities(lastMove)

		confidence := ai.calculatePredictionConfidence(probs)
		weight := ai.adaptiveWeights[order-1] * confidence
		totalWeight += weight

		for i := range weightedPrediction {
			weightedPrediction[i] += probs[i] * weight
		}
	}

	if totalWeight == 0 {
		for i := range weightedPrediction {
			weightedPrediction[i] = 1.0 / MaxMoves
		}
		return weightedPrediction
	}

	for i := range weightedPrediction {
		weightedPrediction[i] /= totalWeight
	}

	return weightedPrediction
}

func (ai *MarkovChainAI) calculatePredictionConfidence(probs [MaxMoves]float64) float64 {
	var entropy float64
	maxProb := 0.0

	for _, prob := range probs {
		if prob > 0 {
			entropy -= prob * math.Log2(prob)
			if prob > maxProb {
				maxProb = prob
			}
		}
	}

	maxEntropy := math.Log2(MaxMoves)
	normalizedEntropy := entropy / maxEntropy

	return maxProb * (1.0 - normalizedEntropy)
}

func (ai *MarkovChainAI) selectOptimalResponse(predictions [MaxMoves]float64, gameState *GameState) Move {
	bestMove := Move(0)
	bestExpectedValue := math.Inf(-1)

	for myMove := Move(0); myMove < MaxMoves; myMove++ {
		var expectedValue float64

		for opponentMove := Move(0); opponentMove < MaxMoves; opponentMove++ {
			probability := predictions[opponentMove]
			result := gameState.EvaluateMove(myMove, opponentMove)
			payoff := gameState.GetPayoff(result)
			expectedValue += probability * payoff
		}

		if expectedValue > bestExpectedValue {
			bestExpectedValue = expectedValue
			bestMove = myMove
		}
	}

	return bestMove
}

type GameEngine struct {
	gameState *GameState
}

func NewGameEngine() *GameEngine {
	return &GameEngine{
		gameState: NewGameState(),
	}
}

func (ge *GameEngine) PlayGame(player1, player2 AIPlayer, history1, history2 *MoveHistory) GameResult {
	move1 := player1.MakeMove(context.Background(), history2, ge.gameState)
	move2 := player2.MakeMove(context.Background(), history1, ge.gameState)

	result := ge.gameState.EvaluateMove(move1, move2)

	history1.Add(move1)
	history2.Add(move2)

	player1.UpdateStrategy(move1, move2)
	player2.UpdateStrategy(move2, move1)

	return result
}

type GameStats struct {
	player1Wins int64
	player2Wins int64
	draws       int64
	totalGames  int64
	startTime   time.Time
}

func NewGameStats() *GameStats {
	return &GameStats{startTime: time.Now()}
}

func (s *GameStats) AddResult(result GameResult) {
	atomic.AddInt64(&s.totalGames, 1)
	switch result {
	case Player1Win:
		atomic.AddInt64(&s.player1Wins, 1)
	case Player2Win:
		atomic.AddInt64(&s.player2Wins, 1)
	case Draw:
		atomic.AddInt64(&s.draws, 1)
	}
}

func (s *GameStats) GetWinRate(player int) float64 {
	total := atomic.LoadInt64(&s.totalGames)
	if total == 0 {
		return 0.0
	}

	switch player {
	case 1:
		return float64(atomic.LoadInt64(&s.player1Wins)) / float64(total)
	case 2:
		return float64(atomic.LoadInt64(&s.player2Wins)) / float64(total)
	}
	return 0.0
}

type MatchupResult struct {
	Player1Type string
	Player2Type string
	Stats       *GameStats
}

type Tournament struct {
	engine           *GameEngine
	workers          int
	games            int
	parallelSessions int
}

func NewTournament(workers, games, parallelSessions int) *Tournament {
	return &Tournament{
		engine:           NewGameEngine(),
		workers:          workers,
		games:            games,
		parallelSessions: parallelSessions,
	}
}

func (t *Tournament) RunSession(ctx context.Context, player1, player2 AIPlayer) *GameStats {
	stats := NewGameStats()
	history1 := NewMoveHistory(2048)
	history2 := NewMoveHistory(2048)

	for i := 0; i < t.games; i++ {
		select {
		case <-ctx.Done():
			return stats
		default:
		}

		result := t.engine.PlayGame(player1, player2, history1, history2)
		stats.AddResult(result)
	}

	return stats
}

func (t *Tournament) Run(ctx context.Context) []MatchupResult {
	aiFactories := map[string]func(string) AIPlayer{
		"minimax": func(name string) AIPlayer { return NewAlphaBetaAI(name, 3) },
		"markov":  func(name string) AIPlayer { return NewMarkovChainAI(name, 4) },
	}

	type workRequest struct {
		player1Type string
		player2Type string
		sessionID   int
	}

	workQueue := make(chan workRequest, t.workers*2)
	resultQueue := make(chan MatchupResult, t.workers*2)

	var wg sync.WaitGroup

	for i := 0; i < t.workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()

			for {
				select {
				case <-ctx.Done():
					return
				case req, ok := <-workQueue:
					if !ok {
						return
					}

					player1 := aiFactories[req.player1Type](fmt.Sprintf("P1_%s_%d", req.player1Type, req.sessionID))
					player2 := aiFactories[req.player2Type](fmt.Sprintf("P2_%s_%d", req.player2Type, req.sessionID))

					stats := t.RunSession(ctx, player1, player2)

					select {
					case resultQueue <- MatchupResult{
						Player1Type: req.player1Type,
						Player2Type: req.player2Type,
						Stats:       stats,
					}:
					case <-ctx.Done():
						return
					}
				}
			}
		}()
	}

	sessionID := 0
	aiTypes := []string{"minimax", "markov"}

	for _, ai1 := range aiTypes {
		for _, ai2 := range aiTypes {
			if ai1 != ai2 {
				for j := 0; j < t.parallelSessions; j++ {
					select {
					case <-ctx.Done():
						close(workQueue)
						goto waitForCompletion
					default:
						workQueue <- workRequest{
							player1Type: ai1,
							player2Type: ai2,
							sessionID:   sessionID,
						}
						sessionID++
					}
				}
			}
		}
	}

	close(workQueue)

waitForCompletion:
	go func() {
		wg.Wait()
		close(resultQueue)
	}()

	var results []MatchupResult
	for result := range resultQueue {
		results = append(results, result)
	}

	return results
}

type SignalManager struct {
	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup
}

func NewSignalManager() *SignalManager {
	ctx, cancel := context.WithCancel(context.Background())
	sm := &SignalManager{ctx: ctx, cancel: cancel}

	signalChan := make(chan os.Signal, 1)
	signal.Notify(signalChan, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		<-signalChan
		sm.cancel()

		done := make(chan struct{})
		go func() {
			sm.wg.Wait()
			close(done)
		}()

		select {
		case <-done:
		case <-time.After(30 * time.Second):
		}
	}()

	return sm
}

func (sm *SignalManager) Context() context.Context {
	return sm.ctx
}

func (sm *SignalManager) Add(delta int) {
	sm.wg.Add(delta)
}

func (sm *SignalManager) Done() {
	sm.wg.Done()
}

type ResultsAnalyzer struct{}

func NewResultsAnalyzer() *ResultsAnalyzer {
	return &ResultsAnalyzer{}
}

func (ra *ResultsAnalyzer) DisplayResults(tournamentResults [][]MatchupResult) {
	overallStats := make(map[string]map[string]*GameStats)

	for _, results := range tournamentResults {
		for _, result := range results {
			if overallStats[result.Player1Type] == nil {
				overallStats[result.Player1Type] = make(map[string]*GameStats)
			}

			if existing := overallStats[result.Player1Type][result.Player2Type]; existing != nil {
				atomic.AddInt64(&existing.player1Wins, atomic.LoadInt64(&result.Stats.player1Wins))
				atomic.AddInt64(&existing.player2Wins, atomic.LoadInt64(&result.Stats.player2Wins))
				atomic.AddInt64(&existing.draws, atomic.LoadInt64(&result.Stats.draws))
				atomic.AddInt64(&existing.totalGames, atomic.LoadInt64(&result.Stats.totalGames))
			} else {
				overallStats[result.Player1Type][result.Player2Type] = result.Stats
			}
		}
	}

	var aiTypes []string
	for aiType := range overallStats {
		aiTypes = append(aiTypes, aiType)
	}
	sort.Strings(aiTypes)

	fmt.Printf("\n%-12s", "AI Type")
	for _, opponent := range aiTypes {
		fmt.Printf("%-15s", opponent)
	}
	fmt.Println()

	for _, aiType := range aiTypes {
		fmt.Printf("%-12s", aiType)
		for _, opponent := range aiTypes {
			if aiType == opponent {
				fmt.Printf("%-15s", "---")
			} else if stat := overallStats[aiType][opponent]; stat != nil {
				winRate := stat.GetWinRate(1) * 100
				fmt.Printf("%-15s", fmt.Sprintf("%.1f%%", winRate))
			} else {
				fmt.Printf("%-15s", "N/A")
			}
		}
		fmt.Println()
	}

	fmt.Printf("\n🏅 Performance Summary:\n")
	for _, aiType := range aiTypes {
		var totalWins, totalGames int64

		for _, opponent := range aiTypes {
			if stat := overallStats[aiType][opponent]; stat != nil {
				totalWins += atomic.LoadInt64(&stat.player1Wins)
				totalGames += atomic.LoadInt64(&stat.totalGames)
			}
		}

		if totalGames > 0 {
			overallWinRate := float64(totalWins) / float64(totalGames) * 100
			fmt.Printf("  %s: %.1f%% (%d/%d)\n", aiType, overallWinRate, totalWins, totalGames)
		}
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

var (
	numGames      = flag.Int("games", 1000, "Number of games per session")
	parallelGames = flag.Int("parallel", 8, "Number of parallel game sessions")
	workers       = flag.Int("workers", runtime.NumCPU(), "Number of worker threads")
	tournaments   = flag.Int("tournaments", 3, "Number of tournament rounds")
)

func main() {
	flag.Parse()

	if *numGames <= 0 || *parallelGames <= 0 || *tournaments <= 0 || *workers <= 0 {
		fmt.Println("Error: All parameters must be positive")
		os.Exit(1)
	}

	sm := NewSignalManager()
	tournament := NewTournament(*workers, *numGames, *parallelGames)
	analyzer := NewResultsAnalyzer()

	fmt.Println("🎮 Minimax Alpha-Beta vs Markov Chain AI Tournament")
	fmt.Printf("Config: %d tournaments, %d parallel, %d workers, %d games\n",
		*tournaments, *parallelGames, *workers, *numGames)

	startTime := time.Now()
	var allResults [][]MatchupResult

	for t := 1; t <= *tournaments; t++ {
		select {
		case <-sm.Context().Done():
			goto shutdown
		default:
		}

		results := tournament.Run(sm.Context())
		if results != nil {
			allResults = append(allResults, results)
		}
	}

shutdown:
	if len(allResults) > 0 {
		analyzer.DisplayResults(allResults)
	}

	fmt.Printf("\n🏁 Completed %d tournaments in %v\n",
		len(allResults), time.Since(startTime).Round(time.Millisecond))
}
