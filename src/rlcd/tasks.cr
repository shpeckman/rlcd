# src/rlcd/tasks.cr
module Rlcd
  class NoulTask
    getter featurizer   : Featurizer
    getter option_count : Int32

    @rng       : Random
    @questions : Array(String)
    @criteria  : Array(String)
    @curated   : Array(Tuple(String, Int32))

    def initialize(@dim : Int32 = 128, @seed : UInt64 = 42_u64, curated : Array(Tuple(String, Int32))? = nil)
      @featurizer   = Featurizer.new(dim: @dim, seed: @seed)
      @option_count = 2
      @rng          = Random.new(@seed)
      @questions = [
        "Should I escalate this ticket to tier-2?",
        "Does this ticket warrant escalation?",
        "Should this issue be routed to tier-2 support?",
      ]
      @criteria = [
        "Escalate to tier-2 when severity is critical, when severity is high and more than 20 users are affected, or when a security or data-loss issue is reported.",
        "Tier-2 policy: escalate on critical severity, on high severity with over 20 affected users, or on any security/data-loss keyword.",
      ]
      @curated = curated || [
        {"Payments are failing for hundreds of users across the EU region", 1},
        {"A single user cannot reset their password",                       0},
        {"Login is completely down for every account since 9am",            1},
        {"One customer reports a typo on the invoice PDF",                  0},
        {"Suspected credential leak reported by the security team",         1},
        {"A few users see slow page loads during peak hours",               0},
        {"The export endpoint returns unencrypted personal data",           1},
        {"Two customers want to change their subscription plan",            0},
        {"All checkouts are timing out for the entire customer base",       1},
        {"One user wants to merge two accounts",                            0},
        {"Database connection errors affect 150 concurrent sessions",       1},
        {"A customer asks about a refund policy",                           0},
        {"Possible data exposure in the public share link",                 1},
        {"Several users report intermittent sync issues",                   0},
        {"The API is returning 500s for every request in production",       1},
        {"An individual account has a duplicate display name",              0},
        {"Privilege escalation was reported on an internal tool",           1},
        {"A user forgot their two-factor backup codes",                     0},
        {"Billing discrepancies affect dozens of corporate accounts",       1},
        {"One customer reports a cosmetic UI glitch",                       0},
        {"Critical outage: payments service down across all regions",       1},
        {"A user would like dark mode enabled by default",                  0},
        {"High-severity memory leak affects 40 long-running workers",       1},
        {"A customer asks how to export their data",                        0},
        {"Sensitive personal data is being logged in plaintext",            1},
        {"Three users report occasional email delivery delays",             0},
        {"Every account in the staging environment is locked out",          1},
        {"A customer wants a copy of their invoice",                        0},
        {"Unencrypted payment data is visible in the admin dashboard",      1},
        {"A user reports a slow dashboard on one old browser",              0},
      ]
    end

    def examples : Array(Example)
      result = Array(Example).new(@curated.size + 1600)
      @curated.each_with_index do |(state, label), i|
        result << make_example(state, @questions[i % @questions.size], @criteria[i % @criteria.size], label)
      end
      generate_templates.each do |(state, label)|
        result << make_example(state, @questions[@rng.rand(@questions.size)], @criteria[@rng.rand(@criteria.size)], label)
      end
      result
    end

    private def make_example(state : String, question : String, criteria : String, label : Int32) : Example
      Example.new(@featurizer.features(state, question, criteria), label)
    end

    private def generate_templates : Array(Tuple(String, Int32))
      result = Array(Tuple(String, Int32)).new(1600)
      sev_words = {
        0 => ["minor", "isolated", "low-impact"],
        1 => ["moderate", "intermittent", "localized"],
        2 => ["major", "widespread", "high-impact"],
        3 => ["critical", "total", "service-wide"],
      }
      issues    = ["payment failures", "login errors", "sync issues", "billing discrepancies", "performance degradation"]
      sec_words = ["possible data exposure", "suspected credential leak", "unencrypted personal data", "privilege escalation report"]

      1600.times do
        severity = @rng.rand(4)
        affected = @rng.rand(1..500)
        security = @rng.rand(2) == 1
        issue    = issues[@rng.rand(issues.size)]
        sev      = sev_words[severity][@rng.rand(3)]
        state =
          if security
            "Reported #{sec_words[@rng.rand(sec_words.size)]}; also #{issue} affecting #{affected} users, severity #{sev}"
          else
            "#{issue} affecting #{affected} users, severity reported as #{sev}"
          end
        label = escalate?(severity, affected, security) ? 1 : 0
        result << {state, label}
      end
      result
    end

    private def escalate?(severity : Int32, affected : Int32, security : Bool) : Bool
      severity >= 3 || (severity == 2 && affected > 20) || security
    end
  end

  class IntentTask
    getter featurizer   : Featurizer
    getter option_count : Int32

    @rng       : Random
    @questions : Array(String)
    @criteria  : Array(String)
    @curated   : Array(Tuple(String, Int32))

    def initialize(@dim : Int32 = 128, @seed : UInt64 = 42_u64, curated : Array(Tuple(String, Int32))? = nil)
      @featurizer   = Featurizer.new(dim: @dim, seed: @seed)
      @option_count = 4
      @rng          = Random.new(@seed)
      @questions = [
        "What is the user's intent?",
        "Which support intent does this message express?",
      ]
      @criteria = [
        "Intents: billing = invoices, charges, payments; refund = money-back, reimbursement; account = login, password, access; technical = crash, bug, malfunction.",
      ]
      @curated = curated || [
        {"I was charged twice for my last invoice",                       0},
        {"Can I get my money back for this order?",                       1},
        {"I cannot log in after resetting my password",                   2},
        {"The app crashes whenever I open the settings page",             3},
        {"My payment was declined but the amount was taken",              0},
        {"I want a refund for the unused portion of my plan",             1},
        {"How do I recover access to my locked account?",                 2},
        {"The export button does nothing on my machine",                  3},
        {"The invoice shows a different amount than I agreed",            0},
        {"Please reimburse the double charge from last week",             1},
        {"My two-factor codes are not arriving",                          2},
        {"The dashboard freezes on large datasets",                       3},
        {"I need a corrected receipt for my company records",             0},
        {"I would like to cancel and get a prorated refund",              1},
        {"I forgot my username and the recovery email is old",            2},
        {"Search returns no results for existing customers",              3},
        {"Why is there an extra tax line on this invoice?",               0},
        {"The product does not work as advertised, I want my money back", 1},
        {"Please help me change the email on my account",                 2},
        {"Uploading a profile picture returns an error",                  3},
        {"My card was charged after I canceled",                          0},
        {"Refund requested for the accidental annual renewal",            1},
        {"I am locked out and the unlock link expired",                   2},
        {"Notifications do not fire for new messages",                    3},
        {"The billing statement is missing my discount",                  0},
        {"Send the reimbursement to my original payment method",          1},
        {"How do I delete my account permanently?",                       2},
        {"The mobile app signs me out every few minutes",                 3},
        {"I need a VAT invoice for the last subscription",                0},
        {"I was overcharged, please return the difference",               1},
        {"Account recovery email is going to an old address",             2},
        {"The reports page shows a blank screen",                         3},
      ]
    end

    def examples : Array(Example)
      result = Array(Example).new(@curated.size + 1600)
      @curated.each_with_index do |(state, label), i|
        result << make_example(state, @questions[i % @questions.size], @criteria[i % @criteria.size], label)
      end
      generate_templates.each do |(state, label)|
        result << make_example(state, @questions[@rng.rand(@questions.size)], @criteria[@rng.rand(@criteria.size)], label)
      end
      result
    end

    private def make_example(state : String, question : String, criteria : String, label : Int32) : Example
      Example.new(@featurizer.features(state, question, criteria), label)
    end

    private def generate_templates : Array(Tuple(String, Int32))
      result = Array(Tuple(String, Int32)).new(1600)
      phrases = {
        0 => ["charged twice", "invoice amount", "payment declined", "billing statement", "extra tax line", "card charged after cancel", "missing discount", "VAT invoice", "corrected receipt"],
        1 => ["money back", "refund for the plan", "reimburse the charge", "prorated refund", "return the difference", "accidental renewal refund", "cancel and refund"],
        2 => ["cannot log in", "reset my password", "locked account", "two-factor codes", "forgot my username", "recovery email", "change the email", "delete my account"],
        3 => ["app crashes", "button does nothing", "dashboard freezes", "search returns nothing", "upload error", "notifications do not fire", "signs me out", "blank screen"],
      }

      1600.times do
        intent = @rng.rand(4)
        phrase = phrases[intent][@rng.rand(phrases[intent].size)]
        result << {"Customer message: #{phrase}", intent}
      end
      result
    end
  end

  def self.split_examples(examples : Array(Example), train_frac : Float64, val_frac : Float64, rng : Random) : {Array(Example), Array(Example), Array(Example)}
    shuffled = examples.shuffle(rng)
    n        = shuffled.size
    t        = (n * train_frac).to_i
    v        = (n * val_frac).to_i
    {shuffled[0, t], shuffled[t, v], shuffled[t + v, n - t - v]}
  end
end
