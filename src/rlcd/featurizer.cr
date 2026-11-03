# src/rlcd/featurizer.cr
require "json"

module RLCD
  record Decision, state : String, question : String, criteria : String

  class Featurizer
    getter  dim  : Int32
    getter  seed : UInt64
    getter? code : Bool

    def initialize(@dim : Int32 = 128, @seed : UInt64 = 0xcbf29ce484222325_u64, @code : Bool = false)
      raise ArgumentError.new("dim must be positive") if @dim <= 0
    end

    def self.new(pull : JSON::PullParser) : Featurizer
      dim  = nil.as(Int32?)
      seed = nil.as(UInt64?)
      code = nil.as(Bool?)
      pull.read_object do |key|
        case key
        when "dim"  then dim = Int32.new(pull)
        when "seed" then seed = UInt64.new(pull)
        when "code" then code = Bool.new(pull)
        else             pull.skip
        end
      end
      checked(
        dim || raise(FormatError.new("featurizer is missing dim")),
        seed || raise(FormatError.new("featurizer is missing seed")),
        code.nil? ? raise(FormatError.new("featurizer is missing code")) : code,
      )
    end

    def self.decode(io : IO::Memory, format : IO::ByteFormat) : Featurizer
      dim  = io.read_bytes(Int32, format)
      seed = io.read_bytes(UInt64, format)
      code = io.read_byte || raise IO::EOFError.new
      raise FormatError.new("featurizer code flag must be 0 or 1, got #{code}") if code > 1
      checked(dim, seed, code == 1)
    end

    private def self.checked(dim : Int32, seed : UInt64, code : Bool) : Featurizer
      raise FormatError.new("featurizer dim must be positive, got #{dim}") if dim <= 0
      new(dim, seed, code)
    end

    def to_json(json : JSON::Builder) : Nil
      json.object do
        json.field "dim", @dim
        json.field "seed", @seed
        json.field "code", @code
      end
    end

    def encode(io : IO, format : IO::ByteFormat) : Nil
      io.write_bytes(@dim, format)
      io.write_bytes(@seed, format)
      io.write_byte(@code ? 1_u8 : 0_u8)
    end

    def features(decision : Decision) : Array(Float64)
      features(decision.state, decision.question, decision.criteria)
    end

    def features(state : String, question : String, criteria : String) : Array(Float64)
      vec = Array(Float64).new(@dim, 0.0)
      tokens_for(state).each { |t| add_token(vec, t, 0_u64) }
      tokens_for(question).each { |t| add_token(vec, t, 1_u64) }
      tokens_for(criteria).each { |t| add_token(vec, t, 2_u64) }
      norm = Math.sqrt(vec.sum { |v| v * v })
      norm > 1e-12 ? vec.map { |v| v / norm } : vec
    end

    def features(state : JSON::Any, question : String, criteria : String) : Array(Float64)
      features(flatten_json(state), resolve_refs(question, state), resolve_refs(criteria, state))
    end

    private def tokens_for(text : String) : Array(String)
      @code ? code_tokenize(text) : tokenize(text)
    end

    private def tokenize(text : String) : Array(String)
      text.downcase.split(/[^a-z0-9]+/).reject { |t| t.size < 2 }
    end

    private def code_tokenize(text : String) : Array(String)
      return [] of String if text.empty?

      normalized = text.downcase.gsub(/\r\n?/, "\n")
      tokens = tokenize(normalized).map { |t| "w:#{t}" }
      normalized.scan(/[^\p{L}\p{N}\s]+/) do |match|
        run = match[0]
        tokens << "r:#{run}"
        run.each_char { |char| tokens << "s:#{char}" }
      end

      chars = "\u0002#{normalized}\u0003".chars
      (2..5).each do |n|
        next if chars.size < n
        (0..chars.size - n).each do |i|
          tokens << "c#{n}:#{chars[i, n].join}"
        end
      end
      tokens
    end

    private def add_token(vec : Array(Float64), token : String, field : UInt64) : Nil
      h = hash_token(field, token)
      idx = (h % @dim.to_u64).to_i
      sign = ((h >> 63) & 1_u64) == 1_u64 ? -1.0 : 1.0
      vec[idx] += sign
    end

    private def hash_token(field : UInt64, token : String) : UInt64
      h = @seed ^ (field &* 0x9e3779b97f4a7c15_u64)
      token.each_byte do |b|
        h ^= b.to_u64
        h &*= 0x100000001b3_u64
      end
      h
    end

    private def flatten_json(node : JSON::Any) : String
      case node.raw
      when String  then node.as_s
      when Int64   then node.as_i.to_s
      when Float64 then node.as_f.to_s
      when Bool    then node.as_bool.to_s
      when Nil     then ""
      when Array   then node.as_a.map { |e| flatten_json(e) }.join(" ")
      when Hash    then node.as_h.map { |k, v| "#{k}: #{flatten_json(v)}" }.join(" ")
      else              node.to_s
      end
    end

    private def resolve_refs(text : String, state : JSON::Any) : String
      builder = String::Builder.new
      pos = 0
      text.scan(/`[^`]+`/) do |m|
        start_i = m.begin(0)
        end_i = m.end(0)
        builder << text[pos...start_i]
        inner = text[(start_i + 1)...(end_i - 1)]
        value = resolve_path(state, inner)
        builder << (value ? flatten_json(value) : text[start_i...end_i])
        pos = end_i
      end
      builder << text[pos..]
      builder.to_s
    end

    private def resolve_path(node : JSON::Any, path : String) : JSON::Any?
      current = node
      path.tr("[]", "..").split('.').reject(&.empty?).each do |key|
        case current.raw
        when Hash
          h = current.as_h
          return nil unless h.has_key?(key)
          current = h[key]
        when Array
          i = key.to_i?
          return nil if i.nil?
          arr = current.as_a
          return nil if i < 0 || i >= arr.size
          current = arr[i]
        else
          return nil
        end
      end
      current
    end
  end
end
