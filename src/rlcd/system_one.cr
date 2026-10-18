# src/rlcd/system_one.cr
require "json"

module RLCD
  enum Primitive
    Noul
    Choice
    Score
  end

  record Question, id : String, primitive : Primitive, instructions : String, criteria : Array(String)

  class SystemOne
    def initialize(@featurizer : Featurizer, @policies : Hash(Int32, Policy))
    end

    def answer(state : String, questions : Array(Question)) : Array(NoulAnswer | ChoiceAnswer | ScoreAnswer)
      questions.map do |q|
        policy = @policies[q.criteria.size]? || raise ArgumentError.new("no policy for #{q.criteria.size} options")
        answer_one(q, policy.probs(@featurizer.features(state, q.instructions, q.criteria.join(" | "))))
      end
    end

    def answer(state : JSON::Any, questions : Array(Question)) : Array(NoulAnswer | ChoiceAnswer | ScoreAnswer)
      questions.map do |q|
        policy = @policies[q.criteria.size]? || raise ArgumentError.new("no policy for #{q.criteria.size} options")
        answer_one(q, policy.probs(@featurizer.features(state, q.instructions, q.criteria.join(" | "))))
      end
    end

    private def answer_one(question : Question, probs : Array(Float64)) : NoulAnswer | ChoiceAnswer | ScoreAnswer
      case question.primitive
      in .noul?
        NoulAnswer.new(question.id, probs[1], probs)
      in .choice?
        ChoiceAnswer.new(question.id, probs.index(probs.max).not_nil!, probs, RLCD.confidence(probs))
      in .score?
        ScoreAnswer.new(question.id, RLCD.expected_score(probs), probs, RLCD.score_confidence(probs), question.criteria)
      end
    end
  end
end
