# src/rlcd/model.cr
require "json"
require "./checkpoint"
require "./container"
require "./featurizer"
require "./format"
require "./optimizer"
require "./persistent"
require "./policy"

module RLCD
  class Model
    include Persistent

    getter featurizer  : Featurizer
    getter policy      : Policy
    getter labels      : Array(String)
    getter temperature : Float64
    getter optimizer   : Adam?
    getter checkpoint  : Checkpoint?

    def initialize(@featurizer : Featurizer, @policy : Policy, @labels : Array(String), @temperature : Float64 = 1.0,
                   @optimizer : Adam? = nil, @checkpoint : Checkpoint? = nil)
      raise ArgumentError.new("featurizer dim #{@featurizer.dim} does not match policy input_dim #{@policy.input_dim}") unless @featurizer.dim == @policy.input_dim
      raise ArgumentError.new("#{@labels.size} labels for #{@policy.option_count} policy options") unless @labels.size == @policy.option_count
      raise ArgumentError.new("temperature must be positive and finite, got #{@temperature}") unless @temperature.finite? && @temperature > 0.0
      if (adam = @optimizer) && !adam.policy.same?(@policy)
        raise ArgumentError.new("optimizer belongs to a different policy")
      end
    end

    def self.new(pull : JSON::PullParser, rng : Random = Random.new) : Model
      featurizer  = nil.as(Featurizer?)
      policy      = nil.as(Policy?)
      labels      = nil.as(Array(String)?)
      temperature = nil.as(Float64?)
      optimizer   = nil.as(String?)
      checkpoint  = nil.as(Checkpoint?)
      pull.read_object do |key|
        case key
        when "featurizer"  then featurizer = Featurizer.new(pull)
        when "policy"      then policy = Policy.new(pull, rng)
        when "labels"      then labels = Array(String).new(pull)
        when "temperature" then temperature = Float64.new(pull)
        when "optimizer"   then optimizer = pull.read_raw
        when "checkpoint"  then checkpoint = Checkpoint.new(pull)
        else                    pull.skip
        end
      end

      policy = policy || raise FormatError.new("model is missing policy")
      new(
        featurizer || raise(FormatError.new("model is missing featurizer")),
        policy,
        labels || raise(FormatError.new("model is missing labels")),
        temperature || raise(FormatError.new("model is missing temperature")),
        optimizer.try { |raw| Adam.new(policy, JSON::PullParser.new(raw)) },
        checkpoint,
      )
    end

    def self.from_io(io : IO, format : IO::ByteFormat = IO::ByteFormat::LittleEndian, rng : Random = Random.new) : Model
      chunks = Container.read(io)
      order  = chunks.format
      policy = chunks.decode(Container::TAG[:policy]) { |payload| Policy.decode(payload, order, rng) }
      new(
        chunks.decode(Container::TAG[:featurizer]) { |payload| Featurizer.decode(payload, order) },
        policy,
        chunks.decode(Container::TAG[:labels]) { |payload| decode_labels(payload, order) },
        chunks.decode(Container::TAG[:temperature]) { |payload| payload.read_bytes(Float64, order) },
        chunks.decode?(Container::TAG[:optimizer]) { |payload| Adam.decode(policy, payload, order) },
        chunks.decode?(Container::TAG[:checkpoint]) { |payload| Checkpoint.decode(payload, order) },
      )
    end

    private def self.decode_labels(io : IO::Memory, format : IO::ByteFormat) : Array(String)
      count = io.read_bytes(Int32, format)
      raise FormatError.new("invalid label count #{count}") unless 0 <= count && count.to_i64 * 4 <= io.size - io.pos
      Array(String).new(count) do
        size = io.read_bytes(Int32, format)
        raise FormatError.new("invalid label length #{size}") unless 0 <= size && size <= io.size - io.pos
        bytes = Bytes.new(size)
        io.read_fully(bytes)
        label = String.new(bytes)
        raise FormatError.new("label is not valid UTF-8") unless label.valid_encoding?
        label
      end
    end

    def probs(state : String | JSON::Any, question : String = "", criteria : String = "") : Array(Float64)
      RLCD.temperature_probs(@policy.forward(@featurizer.features(state, question, criteria)).logits, @temperature)
    end

    def predict(state : String | JSON::Any, question : String = "", criteria : String = "") : Array({String, Float64})
      @labels.zip(probs(state, question, criteria)).sort_by! { |(_, p)| -p }
    end

    def to_json(json : JSON::Builder) : Nil
      json.object do
        json.field "featurizer", @featurizer
        json.field "labels", @labels
        json.field "temperature", @temperature
        json.field "policy", @policy
        if adam = @optimizer
          json.field "optimizer", adam
        end
        if checkpoint = @checkpoint
          json.field "checkpoint", checkpoint
        end
      end
    end

    def to_io(io : IO, format : IO::ByteFormat = IO::ByteFormat::LittleEndian, precision : Precision = Precision::F64) : Nil
      Container.write(io, format) do |writer|
        writer.add(Container::TAG[:featurizer]) { |payload| @featurizer.encode(payload, format) }
        writer.add(Container::TAG[:labels]) do |payload|
          payload.write_bytes(@labels.size, format)
          @labels.each do |label|
            payload.write_bytes(label.bytesize, format)
            payload.write(label.to_slice)
          end
        end
        writer.add(Container::TAG[:temperature]) { |payload| payload.write_bytes(@temperature, format) }
        writer.add(Container::TAG[:policy]) { |payload| @policy.encode(payload, format, precision) }
        if adam = @optimizer
          writer.add(Container::TAG[:optimizer]) { |payload| adam.encode(payload, format, precision) }
        end
        if checkpoint = @checkpoint
          writer.add(Container::TAG[:checkpoint]) { |payload| checkpoint.encode(payload, format) }
        end
      end
    end
  end
end
