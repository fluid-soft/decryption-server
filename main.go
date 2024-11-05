package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/tls"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/mux"
	"golang.org/x/net/http2"
)

const (
	CLIENT_API_MODEL_ENV          = "CLIENT_API_MODEL" // public model exposed to clients
	INTERNAL_MODEL_ENV            = "INTERNAL_MODEL"   // model used by LLM provider
	MAX_TOKENS_ENV                = "MAX_TOKENS"
	MAX_NON_STREAM_TOKENS_ENV     = "MAX_NON_STREAM_TOKENS"
	PORT_ENV                      = "PORT"
	LLM_BACKEND_URL_ENV           = "LLM_BACKEND_URL"
	LOAD_BALANCER_URL_ENV         = "LOAD_BALANCER_URL"
	INSTANCE_ID_ENV               = "INSTANCE_ID"
	DEBUG_MODE_ENV                = "DEBUG_MODE"
	LLM_BACKEND_AUTH_KEY_ENV      = "LLM_BACKEND_AUTH_KEY"
	TLS_CERT_PATH_ENV             = "TLS_CERT_PATH"
	TLS_KEY_PATH_ENV              = "TLS_KEY_PATH"
	LOAD_BALANCER_SERVER_NAME_ENV = "LOAD_BALANCER_SERVER_NAME"
	COMPLETION_TIMEOUT_SEC_ENV    = "COMPLETION_TIMEOUT_SEC"
)

type Server struct {
	currentKey             KeyPair
	keyMutex               sync.RWMutex
	llmBackendURL          string
	loadBalancerURL        string
	instanceID             string
	maxTokens              int
	maxNonStreamTokens     int
	supportedModel         string
	internalModel          string
	debugMode              bool
	llmBackendAuthKey      string
	loadBalancerServerName string
	completionTimeoutSec   time.Duration
}

type ChatCompletionRequest struct {
	Model         string         `json:"model"`
	Messages      []ChatMessage  `json:"messages"`
	Stream        bool           `json:"stream"`
	MaxTokens     int            `json:"max_tokens"`
	LogProbs      *int           `json:"logprobs,omitempty"`
	StreamOptions *StreamOptions `json:"stream_options,omitempty"`
}

type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ChatCompletionResponse struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
	Usage   Usage    `json:"usage"`
}

type ChatCompletionChunk struct {
	ID      string        `json:"id"`
	Object  string        `json:"object"`
	Created int64         `json:"created"`
	Model   string        `json:"model"`
	Choices []ChunkChoice `json:"choices"`
}

type Choice struct {
	Index        int         `json:"index"`
	Message      ChatMessage `json:"message"`
	FinishReason string      `json:"finish_reason"`
}

type ChunkChoice struct {
	Index        int              `json:"index"`
	Delta        ChatMessageDelta `json:"delta"`
	FinishReason *string          `json:"finish_reason"`
}

type ChatMessageDelta struct {
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
}

type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type StreamOptions struct {
	IncludeUsage *bool `json:"include_usage,omitempty"`
}

type responseWriter struct {
	http.ResponseWriter
	status   int
	errorMsg string
}

func (rw *responseWriter) WriteHeader(code int) {
	rw.status = code
	rw.ResponseWriter.WriteHeader(code)
}

func (rw *responseWriter) Write(b []byte) (int, error) {
	return rw.ResponseWriter.Write(b)
}

func (rw *responseWriter) WriteError(err string, code int) {
	rw.errorMsg = err
	rw.status = code
	http.Error(rw.ResponseWriter, err, code)
}

func (rw *responseWriter) Flush() {
	if f, ok := rw.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

func loggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()

		recorder := &responseWriter{
			ResponseWriter: w,
			status:         200,
		}

		next.ServeHTTP(recorder, r)

		duration := time.Since(start)

		clientIP := r.Header.Get("X-Forwarded-For")
		if clientIP == "" {
			clientIP = r.RemoteAddr
		}

		ipParts := strings.Split(clientIP, ".")
		if len(ipParts) == 4 {
			ipParts[3] = "*"
			clientIP = strings.Join(ipParts, ".")
		}

		userAgent := r.Header.Get("User-Agent")
		if userAgent == "" {
			userAgent = "Unknown"
		}

		errorContent := ""
		if recorder.errorMsg != "" {
			errorContent = recorder.errorMsg
			if len(errorContent) > 100 {
				errorContent = errorContent[:97] + "..."
			}
			errorContent = " | " + strings.TrimSpace(errorContent)
		}

		// this will log the IP address of a load balancer. It's also anonymized (last octet removed), just in case.
		log.Printf("[INFO] [%d] %s %s | %dms | %s | %s%s",
			recorder.status,
			r.Method,
			r.URL.Path,
			duration.Milliseconds(),
			clientIP,
			userAgent,
			errorContent,
		)
	})
}

func (s *Server) handleNotFound(w http.ResponseWriter, r *http.Request) {
	rw, ok := w.(*responseWriter)
	if !ok {
		rw = &responseWriter{ResponseWriter: w, status: http.StatusNotFound}
	}
	rw.WriteError("Not Found", http.StatusNotFound)
}

func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	jsonResponse := fmt.Sprintf(`{
        "data": [
            {
                "id": "%s",
                "object": "model",
                "created": %d,
                "owned_by": "Fluid"
            }
        ]
    }`, s.supportedModel, time.Now().Unix())

	w.Header().Set("Content-Type", "application/json")
	_, err := w.Write([]byte(jsonResponse))
	if err != nil {
		log.Printf("[ERROR] Failed to write models response: %v", err)
		http.Error(w, "Failed to send models response", http.StatusInternalServerError)
	}
}

func main() {
	log.SetOutput(os.Stdout)
	log.SetFlags(log.Ldate | log.Ltime | log.Lshortfile)

	log.Println("Starting server...")

	llmBackendURL := strings.TrimSuffix(os.Getenv(LLM_BACKEND_URL_ENV), "/")
	if llmBackendURL == "" {
		log.Fatal(LLM_BACKEND_URL_ENV + " environment variable is not set")
	}

	loadBalancerURL := os.Getenv(LOAD_BALANCER_URL_ENV)
	if loadBalancerURL == "" {
		log.Fatal(LOAD_BALANCER_URL_ENV + " environment variable is not set")
	}

	instanceID := os.Getenv(INSTANCE_ID_ENV)
	if instanceID == "" {
		log.Fatal(INSTANCE_ID_ENV + " environment variable is not set")
	}

	maxTokens, err := strconv.Atoi(os.Getenv(MAX_TOKENS_ENV))
	if err != nil || maxTokens <= 0 {
		log.Printf("Invalid or missing %s, using default value of 2048", MAX_TOKENS_ENV)
		maxTokens = 2048
	}

	maxNonStreamTokens, err := strconv.Atoi(os.Getenv(MAX_NON_STREAM_TOKENS_ENV))
	if err != nil || maxNonStreamTokens <= 0 {
		log.Printf("Invalid or missing %s, using default value of 2048", MAX_NON_STREAM_TOKENS_ENV)
		maxNonStreamTokens = 256
	}

	supportedModel := os.Getenv(CLIENT_API_MODEL_ENV)
	if supportedModel == "" {
		log.Fatal(CLIENT_API_MODEL_ENV + " environment variable is not set")
	}

	internalModel := os.Getenv(INTERNAL_MODEL_ENV)
	if internalModel == "" {
		log.Fatal(INTERNAL_MODEL_ENV + " environment variable is not set")
	}

	debugMode, _ := strconv.ParseBool(os.Getenv(DEBUG_MODE_ENV))

	llmBackendAuthKey := os.Getenv(LLM_BACKEND_AUTH_KEY_ENV)
	if llmBackendAuthKey == "" {
		log.Fatal(LLM_BACKEND_AUTH_KEY_ENV + " environment variable is not set")
	}

	loadBalancerServerName := os.Getenv(LOAD_BALANCER_SERVER_NAME_ENV)
	if loadBalancerServerName == "" {
		log.Printf("Warning: %s environment variable is not set. TLS verification may fail.", LOAD_BALANCER_SERVER_NAME_ENV)
	}

	completionTimeoutSecInt, err := strconv.Atoi(os.Getenv(COMPLETION_TIMEOUT_SEC_ENV))
	if err != nil || completionTimeoutSecInt <= 0 {
		completionTimeoutSecInt = 30
		log.Printf("Warning: %s environment variable is not set. Defaulting to %d seconds.", COMPLETION_TIMEOUT_SEC_ENV, completionTimeoutSecInt)
	}

	log.Printf("Server configuration: LLM backend URL: %s, Load Balancer URL: %s, Instance ID: %s",
		llmBackendURL, loadBalancerURL, instanceID)
	log.Printf("Server configuration: Max Tokens: %d, Supported Model: %s, Internal Model: %s",
		maxTokens, supportedModel, internalModel)

	server := &Server{
		llmBackendURL:          llmBackendURL,
		loadBalancerURL:        loadBalancerURL,
		instanceID:             instanceID,
		maxTokens:              maxTokens,
		maxNonStreamTokens:     maxNonStreamTokens,
		supportedModel:         supportedModel,
		internalModel:          internalModel,
		debugMode:              debugMode,
		llmBackendAuthKey:      llmBackendAuthKey,
		loadBalancerServerName: loadBalancerServerName,
		completionTimeoutSec:   time.Duration(completionTimeoutSecInt) * time.Second,
	}

	server.generateNewKey()
	go server.keyRotationScheduler()

	r := mux.NewRouter()

	r.Use(loggingMiddleware)

	r.HandleFunc("/v1/chat/completions", server.handleChatCompletions).Methods("POST")
	r.HandleFunc("/v1/models", server.handleModels).Methods("GET")
	r.HandleFunc("/public-key", server.getPublicKey).Methods("GET")

	notFoundHandler := loggingMiddleware(http.HandlerFunc(server.handleNotFound))
	r.NotFoundHandler = notFoundHandler

	port := os.Getenv(PORT_ENV)
	if port == "" {
		log.Fatal("PORT environment variable is not set")
	}

	srv := &http.Server{
		Addr:    ":" + port,
		Handler: r,
	}

	err = http2.ConfigureServer(srv, &http2.Server{})
	if err != nil {
		log.Fatal("HTTP/2 server can't be started")
	}

	if debugMode {
		log.Printf("Server starting in DEBUG mode on port %s with HTTP/2 support", port)
		log.Fatal(srv.ListenAndServe())
	} else {
		tlsCertPath := os.Getenv(TLS_CERT_PATH_ENV)
		tlsKeyPath := os.Getenv(TLS_KEY_PATH_ENV)

		if tlsCertPath == "" || tlsKeyPath == "" {
			log.Fatal("TLS_CERT_PATH and TLS_KEY_PATH must be set when not in DEBUG mode")
		}

		log.Printf("Server starting with TLS on port %s with HTTP/2 support", port)
		log.Fatal(srv.ListenAndServeTLS(tlsCertPath, tlsKeyPath))
	}
}

func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	rw, ok := w.(*responseWriter)
	if !ok {
		log.Println("ResponseWriter is not of type *responseWriter")
		http.Error(w, "Internal server error", http.StatusInternalServerError)
		return
	}

	var chatRequest ChatCompletionRequest
	var secret []byte
	var err error
	var requestBody []byte

	clientPublicKey := r.Header.Get("X-Client-Public-Key")
	if clientPublicKey != "" {
		encryptedData, err := io.ReadAll(r.Body)
		if err != nil {
			rw.WriteError("Failed to read encrypted request body", http.StatusBadRequest)
			return
		}

		var clientPublicKeys map[string]string
		err = json.Unmarshal([]byte(clientPublicKey), &clientPublicKeys)
		if err != nil {
			rw.WriteError("Invalid X-Client-Public-Key header", http.StatusBadRequest)
			return
		}

		requestBody, secret, err = s.decryptAndDecompressMessage(encryptedData, clientPublicKeys)
		if err != nil {
			s.handleDecryptionError(rw, err)
			return
		}
	} else {
		requestBody, err = io.ReadAll(r.Body)
		if err != nil {
			rw.WriteError("Failed to read request body", http.StatusBadRequest)
			return
		}
	}

	err = json.Unmarshal(requestBody, &chatRequest)
	if err != nil {
		rw.WriteError("Invalid request format", http.StatusBadRequest)
		return
	}

	if chatRequest.Model != s.supportedModel {
		rw.WriteError(fmt.Sprintf("Unsupported model. Only %s is supported.", s.supportedModel), http.StatusBadRequest)
		return
	}

	chatRequest.Model = s.internalModel
	if chatRequest.Stream {
		chatRequest.MaxTokens = s.maxTokens
	} else {
		chatRequest.MaxTokens = s.maxNonStreamTokens
	}

	if len(chatRequest.Messages) == 0 {
		rw.WriteError("Messages array is empty", http.StatusBadRequest)
		return
	}

	llmBackendRequest, err := json.Marshal(chatRequest)
	if err != nil {
		rw.WriteError("Failed to prepare request for LLM Backend", http.StatusBadRequest)
		return
	}

	backendURL := s.llmBackendURL + "/v1/chat/completions"

	req, err := http.NewRequest("POST", backendURL, bytes.NewReader(llmBackendRequest))
	if err != nil {
		rw.WriteError("Failed to create request to LLM Backend", http.StatusBadRequest)
		return
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+s.llmBackendAuthKey)

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		log.Printf("Failed to send request to LLM Backend: %s", err)
		rw.WriteError("Error forwarding request to LLM Backend", http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, err := io.ReadAll(resp.Body)
		if err != nil {
			log.Printf("Failed to read error response body: %v", err)
			rw.WriteError("Internal server error", http.StatusInternalServerError)
			return
		}

		log.Printf("Backend error response (status %d): %s", resp.StatusCode, string(body))

		mappedStatus, mappedMessage := mapBackendError(resp.StatusCode, body)
		rw.WriteError(mappedMessage, mappedStatus)
		return
	}

	if chatRequest.Stream {
		s.handleStreamingResponse(w, resp, secret, clientPublicKey != "")
	} else {
		s.handleNonStreamingResponse(w, resp, secret, clientPublicKey != "")
	}
}

func (s *Server) handleStreamingResponse(w http.ResponseWriter, resp *http.Response, secret []byte, isEncrypted bool) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "Streaming unsupported", http.StatusInternalServerError)
		return
	}

	reader := bufio.NewReader(resp.Body)

	ctx, cancel := context.WithTimeout(context.Background(), s.completionTimeoutSec)
	line, err := s.readLineWithTimeout(ctx, reader)
	cancel()
	if err != nil {
		if errors.Is(err, context.DeadlineExceeded) {
			http.Error(w, "Request timeout", http.StatusGatewayTimeout)
		} else {
			log.Printf("Error reading from LLM Backend: %v", err)
			http.Error(w, "Internal server error", http.StatusInternalServerError)
		}
		return
	}

	trimmedLine := bytes.TrimSpace(line)
	if len(trimmedLine) == 0 || !bytes.HasPrefix(trimmedLine, []byte("data: ")) {
		log.Printf("Invalid chunk format from LLM backend: missing 'data: ' prefix")
		http.Error(w, "Invalid response format from LLM backend", http.StatusInternalServerError)
		return
	}

	// we defer setting SSE headers just in case it error'd in first line
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	if bytes.Equal(trimmedLine, []byte("data: [DONE]")) {
		s.writeStreamMessage(w, flusher, trimmedLine, secret, isEncrypted)
		return
	}

	if err := s.processChunk(trimmedLine, w, flusher, secret, isEncrypted); err != nil {
		return
	}

	for {
		ctx, cancel := context.WithTimeout(context.Background(), s.completionTimeoutSec)
		line, err := s.readLineWithTimeout(ctx, reader)
		cancel()
		if err != nil {
			if errors.Is(err, context.DeadlineExceeded) {
				log.Printf("Timeout reading from LLM Backend")
				s.writeStreamError(w, flusher, "Request timeout", secret, isEncrypted)
			} else if err != io.EOF {
				log.Printf("Error reading from LLM Backend: %v", err)
				s.writeStreamError(w, flusher, "Internal server error", secret, isEncrypted)
			}
			return
		}

		trimmedLine := bytes.TrimSpace(line)
		if len(trimmedLine) == 0 {
			continue
		}

		if bytes.Equal(trimmedLine, []byte("data: [DONE]")) {
			s.writeStreamMessage(w, flusher, trimmedLine, secret, isEncrypted)
			return
		}

		if err := s.processChunk(trimmedLine, w, flusher, secret, isEncrypted); err != nil {
			return
		}
	}
}

func (s *Server) readLineWithTimeout(ctx context.Context, reader *bufio.Reader) ([]byte, error) {
	readChan := make(chan readResult)
	go func() {
		line, err := reader.ReadBytes('\n')
		readChan <- readResult{line: line, err: err}
	}()

	select {
	case <-ctx.Done():
		return nil, context.DeadlineExceeded
	case result := <-readChan:
		return result.line, result.err
	}
}

type readResult struct {
	line []byte
	err  error
}

func (s *Server) processChunk(line []byte, w http.ResponseWriter, flusher http.Flusher, secret []byte, isEncrypted bool) error {
	if !bytes.HasPrefix(line, []byte("data: ")) {
		log.Printf("Invalid chunk format from LLM backend: missing 'data: ' prefix")
		s.writeStreamError(w, flusher, "Invalid response format from LLM backend", secret, isEncrypted)
		return fmt.Errorf("invalid chunk format")
	}

	trimmedLine := bytes.TrimPrefix(line, []byte("data: "))

	var chunk ChatCompletionChunk
	if err := json.Unmarshal(trimmedLine, &chunk); err != nil {
		log.Printf("Invalid chunk format from LLM backend: %v", err)
		s.writeStreamError(w, flusher, "Invalid response format from LLM backend", secret, isEncrypted)
		return err
	}

	chunk.Model = s.supportedModel

	processedChunk, err := json.Marshal(chunk)
	if err != nil {
		log.Printf("Can't marshall chunk: %v", err)
		s.writeStreamError(w, flusher, "Internal server error", secret, isEncrypted)
		return err
	}

	processedChunk = append([]byte("data: "), processedChunk...)
	s.writeStreamMessage(w, flusher, processedChunk, secret, isEncrypted)

	if _, err = w.Write([]byte("\n\n")); err != nil {
		log.Printf("Error writing newline: %v", err)
		s.writeStreamError(w, flusher, "Internal server error", secret, isEncrypted)
		return err
	}
	flusher.Flush()

	return nil
}

func (s *Server) handleNonStreamingResponse(w http.ResponseWriter, resp *http.Response, secret []byte, isEncrypted bool) {
	llmBackendResp, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Printf("Error reading response from LLM Backend: %v", err)
		http.Error(w, "Failed to read response from LLM Backend", http.StatusInternalServerError)
		return
	}

	var response ChatCompletionResponse
	err = json.Unmarshal(llmBackendResp, &response)
	if err != nil {
		log.Printf("Failed to parse LLM backend response: %v", err)
		http.Error(w, "Invalid response format from LLM backend", http.StatusInternalServerError)
		return
	}

	response.Model = s.supportedModel

	processedResp, err := json.Marshal(response)
	if err != nil {
		http.Error(w, "Failed to encode response", http.StatusInternalServerError)
		return
	}

	if isEncrypted {
		encryptedResp, err := s.compressAndEncryptMessage(processedResp, secret)
		if err != nil {
			http.Error(w, "Failed to encrypt response", http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/octet-stream")
		log.Printf("Sending encrypted non-streaming response (length: %d bytes)", len(encryptedResp))
		_, err = w.Write(encryptedResp)
	} else {
		w.Header().Set("Content-Type", "application/json")
		log.Printf("Sending unencrypted non-streaming response (length: %d bytes)", len(processedResp))
		_, err = w.Write(processedResp)
	}

	if err != nil {
		log.Printf("Failed to write response: %v", err)
		return
	}
}

func (s *Server) decryptAndDecompressMessage(encryptedData []byte, clientPublicKeys map[string]string) ([]byte, []byte, error) {
	s.keyMutex.RLock()
	defer s.keyMutex.RUnlock()

	clientPublicKeyBase64, ok := clientPublicKeys[s.instanceID]
	if !ok {
		return nil, nil, errors.New("no public key found for this instance")
	}

	clientPublicKeyBytes, err := base64.StdEncoding.DecodeString(clientPublicKeyBase64)
	if err != nil {
		return nil, nil, err
	}

	symmetricKey, err := DeriveSharedSecret(s.currentKey.PrivateKey, clientPublicKeyBytes)
	if err != nil {
		return nil, nil, err
	}

	payload, err := DecryptAndDecompressAES(encryptedData, symmetricKey)
	if err != nil {
		return nil, nil, err
	}

	return payload, symmetricKey, nil
}

func (s *Server) compressAndEncryptMessage(data, secret []byte) ([]byte, error) {
	return CompressAndEncryptAES(data, secret)
}

func (s *Server) getPublicKey(w http.ResponseWriter, r *http.Request) {
	s.keyMutex.RLock()
	defer s.keyMutex.RUnlock()
	w.Header().Set("Content-Type", "application/json")
	err := json.NewEncoder(w).Encode(map[string]interface{}{
		"publicKey": s.getEncodedPublicKey(),
		"expiresAt": s.currentKey.ExpiresAt,
	})
	if err != nil {
		log.Printf("Failed to encode public key response: %v", err)
		return
	}
}

func (s *Server) getEncodedPublicKey() string {
	return GetEncodedPublicKey(s.currentKey.PublicKey)
}

func (s *Server) generateNewKey() {
	log.Println("Generating new key pair")
	newKeyPair, err := GenerateNewKeyPair()
	if err != nil {
		log.Fatalf("Failed to generate new key pair: %v", err)
	}

	s.keyMutex.Lock()
	defer s.keyMutex.Unlock()

	s.currentKey = *newKeyPair

	log.Println("New key pair generated and set")

	go s.notifyLoadBalancer()
}

func (s *Server) notifyLoadBalancer() {
	log.Println("Notifying load balancer of key update")

	tr := &http.Transport{
		TLSClientConfig: &tls.Config{
			MinVersion: tls.VersionTLS12,
			ServerName: s.loadBalancerServerName,
		},
	}

	client := &http.Client{Transport: tr}

	maxRetryDelay := time.Minute
	minRetryDelay := 2 * time.Second

	retryDelay := time.Millisecond
	effectiveRetryDelay := minRetryDelay

	for {
		resp, err := client.Get(s.loadBalancerURL + "/update-key")
		if err != nil {
			log.Printf("Failed to notify load balancer: %v. Retrying in %v", err, effectiveRetryDelay)
		} else {
			defer resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				log.Println("Load balancer successfully notified")
				return
			}
			log.Printf("Load balancer returned non-OK status: %d. Retrying in %v", resp.StatusCode, effectiveRetryDelay)
		}

		time.Sleep(effectiveRetryDelay)

		// increase the retry delay, but cap it at maxRetryDelay
		retryDelay = time.Duration(math.Min(float64(retryDelay*2), float64(maxRetryDelay)))
		// make sure delay is no smaller than minRetryDelay
		effectiveRetryDelay = time.Duration(math.Max(float64(retryDelay), float64(minRetryDelay)))
	}
}

func (s *Server) keyRotationScheduler() {
	for {
		now := time.Now().UTC()
		nextMidnight := now.Add(24 * time.Hour).Truncate(24 * time.Hour)
		sleepDuration := nextMidnight.Sub(now)

		log.Printf("Next key rotation scheduled for %v (in %v)", nextMidnight, sleepDuration)
		time.Sleep(sleepDuration)

		s.generateNewKey()
		log.Println("Key rotated")
	}
}

func (s *Server) handleDecryptionError(w http.ResponseWriter, err error) {
	s.keyMutex.RLock()
	defer s.keyMutex.RUnlock()

	log.Printf("Decryption error: %v", err)

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusPreconditionFailed)
	response := map[string]interface{}{
		"error":     "Decryption failed. Your public key may be outdated.",
		"publicKey": s.getEncodedPublicKey(),
		"expiresAt": s.currentKey.ExpiresAt,
	}
	log.Printf("Sending error response: %s", response["error"])
	err = json.NewEncoder(w).Encode(response)
	if err != nil {
		log.Printf("Failed to encode response: %v", err)
		return
	}
}

func mapBackendError(statusCode int, body []byte) (int, string) {
	var errorResponse struct {
		Error struct {
			Message string `json:"message"`
			Type    string `json:"type"`
		} `json:"error"`
	}

	errorMessage := "Internal server error"
	if err := json.Unmarshal(body, &errorResponse); err == nil && errorResponse.Error.Message != "" {
		errorMessage = errorResponse.Error.Message
	}

	switch statusCode {
	case http.StatusBadRequest: // 400
		return http.StatusBadRequest, errorMessage
	case http.StatusUnauthorized: // 401
		return http.StatusServiceUnavailable, "Service temporarily unavailable"
	case http.StatusForbidden: // 403
		return http.StatusServiceUnavailable, "Service temporarily unavailable"
	case http.StatusTooManyRequests: // 429
		return http.StatusTooManyRequests, "Too many requests. Please try again later"
	case http.StatusInternalServerError: // 500
		return http.StatusServiceUnavailable, "Service temporarily unavailable"
	case http.StatusBadGateway: // 502
		return http.StatusServiceUnavailable, "Service temporarily unavailable"
	case http.StatusServiceUnavailable: // 503
		return http.StatusServiceUnavailable, "Service temporarily unavailable"
	case http.StatusGatewayTimeout: // 504
		return http.StatusGatewayTimeout, "Request timeout"
	default:
		return http.StatusInternalServerError, "Service temporarily unavailable"
	}
}

func (s *Server) writeStreamError(w http.ResponseWriter, flusher http.Flusher, message string, secret []byte, isEncrypted bool) {
	errorMsg := fmt.Sprintf("data: {\"error\": {\"message\": \"%s\", \"type\": \"server_error\"}}\n\n", message)
	s.writeStreamMessage(w, flusher, []byte(errorMsg), secret, isEncrypted)
}

func (s *Server) writeStreamMessage(w http.ResponseWriter, flusher http.Flusher, message []byte, secret []byte, isEncrypted bool) {
	if isEncrypted {
		encryptedMsg, err := s.compressAndEncryptMessage(message, secret)
		if err != nil {
			log.Printf("Failed to encrypt message: %v", err)
			return
		}
		_, _ = w.Write([]byte(base64.StdEncoding.EncodeToString(encryptedMsg)))
	} else {
		_, _ = w.Write(message)
	}
	flusher.Flush()
}
