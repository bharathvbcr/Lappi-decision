//! Source (c): code in languages the mutation pool never held, authored here.
//!
//! The `code.defect_class` corpus is python, go, rust and typescript; the OOD suite's
//! unseen-language cases are c, java, ruby, haskell, sql and lua (`qd_train.ood._UNSEEN`). These
//! templates are kotlin, csharp, php, scala, elixir and shell -- in neither set -- and their
//! identifiers are drawn from pools that share no name with the suite's (`_NAMES`, `_FNS`) and
//! avoid its motifs (items, weights, active flags, totals, summaries). The disjointness is
//! asserted against the suite's own constants by `python/tests/test_defect_noul.py`, not by a
//! restated copy here.
//!
//! **One split unit per template.** Instantiations of one template sit at Jaccard >= 0.8 on
//! 5-token shingles; under per-row repo keys the dedupe pass would chain them across "repos" and
//! drop all but one. A template is the unit near-duplicates share, so it is the unit the split
//! assigns -- `noul-template/<language>/<name>`.
//!
//! Licence: authored in this repository and released under its `LICENSE` (Apache-2.0, the
//! workspace `license` in `Cargo.toml`). The allowlist step carries that id in, and the loader
//! refuses a template row that claims any other.

use std::collections::{BTreeMap, BTreeSet};

use rand::Rng;
use rand::seq::IndexedRandom;
use rand_chacha::ChaCha20Rng;
use sha2::{Digest, Sha256};

use crate::hunk::{self, item_rng, Line, Mark};

pub const SOURCE: &str = "unseen-language";
pub const UNIT_PREFIX: &str = "noul-template/";

pub struct Template {
    pub language: &'static str,
    pub name: &'static str,
    pub path: &'static str,
    /// One diff line each: a marker (` `, `-`, `+`) then the text, with `«key»` placeholders.
    pub lines: &'static [&'static str],
}

impl Template {
    pub fn unit(&self) -> String {
        format!("{UNIT_PREFIX}{}/{}", self.language, self.name)
    }
}

const TYPES: &[&str] = &[
    "Invoice", "Shipment", "Tenant", "Voucher", "Parcel", "Courier", "Warehouse", "Ticket",
    "Coupon", "Subscriber", "Playlist", "Thermostat", "Greenhouse", "Turbine", "Orchard",
    "Lantern", "Harbor", "Beacon", "Ferry", "Pantry", "Quarry", "Meadow", "Kiln", "Glacier",
];
const VERBS: &[&str] = &[
    "archive_stale", "refresh_quota", "assign_courier", "apply_coupon", "rotate_keys",
    "prune_cache", "export_ledger", "reconcile_balance", "book_pickup", "throttle_bursts",
    "renew_lease", "flag_overdue", "merge_profiles", "normalize_postcode", "recompute_tariff",
    "drain_backlog", "seal_batch", "warm_cache",
];
const NOUNS: &[&str] = &[
    "due_date", "postcode", "retry_limit", "grace_days", "tracking_code", "region_code",
    "batch_size", "unit_price", "tax_rate", "max_attempts", "timeout_ms", "label_text",
    "door_number", "shelf_slot", "humidity", "voltage", "latitude", "billing_email", "locale",
    "currency",
];
const WORDS: &[&str] = &[
    "pending", "shipped", "expired", "north", "amber", "sealed", "draft", "queued", "paused",
    "retired", "frozen", "partial",
];
const ORGS: &[&str] = &["northwind", "brightline", "tidewater", "copperleaf", "larkspur", "ironbark"];

pub const CATALOGUE: &[Template] = &[
    // -- kotlin ----------------------------------------------------------------------------
    Template { language: "kotlin", name: "repository-cache",
        path: "app/src/main/kotlin/com/«org»/«t»/«T»Repository.kt",
        lines: &[
            " package com.«org».«t»",
            " ",
            " class «T»Repository(private val api: «T»Api) {",
            "     private val cache = mutableMapOf<String, «T»>()",
            " ",
            "-    suspend fun «f»(key: String): «T» = api.load(key)",
            "+    suspend fun «f»(key: String): «T» =",
            "+        cache.getOrPut(key) { api.load(key) }",
            " ",
            "     fun clear() = cache.clear()",
            " }",
        ] },
    Template { language: "kotlin", name: "data-class-default",
        path: "core/src/main/kotlin/com/«org»/model/«T».kt",
        lines: &[
            " import java.time.Duration",
            " import java.time.Instant",
            " ",
            " data class «T»(",
            "     val «a»: String,",
            "-    val «b»: Int,",
            "+    val «b»: Int = «n»,",
            "+    val createdAt: Instant = Instant.now(),",
            " ) {",
            "-    fun label(): String = «a».uppercase()",
            "+    fun isStale(now: Instant): Boolean =",
            "+        Duration.between(createdAt, now).toDays() > «n»",
            " }",
        ] },
    Template { language: "kotlin", name: "when-branch",
        path: "app/src/main/kotlin/com/«org»/«t»/«T»Labels.kt",
        lines: &[
            " package com.«org».«t»",
            " ",
            " fun «f»(state: «T»State): String = when (state) {",
            "     «T»State.Pending -> \"«w»\"",
            "     «T»State.Done -> \"done\"",
            "-    else -> \"unknown\"",
            "+    «T»State.Cancelled -> \"cancelled\"",
            "+    else -> error(\"unexpected state $state\")",
            " }",
        ] },
    Template { language: "kotlin", name: "retry-backoff",
        path: "net/src/main/kotlin/com/«org»/net/«T»Retry.kt",
        lines: &[
            " suspend fun <R> «f»(maxRetries: Int = «n», block: suspend () -> R): R {",
            "-    repeat(maxRetries) {",
            "+    repeat(maxRetries) { attempt ->",
            "         try {",
            "             return block()",
            "         } catch (e: IOException) {",
            "-            delay(«m»)",
            "+            delay(«m»L * (attempt + 1))",
            "         }",
            "     }",
            "     return block()",
            " }",
        ] },
    Template { language: "kotlin", name: "extension-check",
        path: "app/src/main/kotlin/com/«org»/«t»/«T»Extensions.kt",
        lines: &[
            " package com.«org».«t»",
            " ",
            "-fun «T».«f»(): Boolean = «a».isNotBlank()",
            "+fun «T».«f»(): Boolean =",
            "+    «a».isNotBlank() && «b» > «n»",
            " ",
            " fun List<«T»>.readyCount(): Int = count { it.«f»() }",
            " ",
            " fun «T».describe(): String = \"«T»($«a»)\"",
        ] },
    Template { language: "kotlin", name: "sealed-result",
        path: "domain/src/main/kotlin/com/«org»/«t»/«T»Result.kt",
        lines: &[
            " package com.«org».«t»",
            " ",
            " sealed interface «T»Result {",
            "     data class Ok(val «a»: String) : «T»Result",
            "-    object Failed : «T»Result",
            "+    data class Failed(val reason: String) : «T»Result",
            "+    data object Skipped : «T»Result",
            " }",
            " ",
            " fun «T»Result.isOk(): Boolean = this is «T»Result.Ok",
        ] },
    Template { language: "kotlin", name: "state-flow",
        path: "app/src/main/kotlin/com/«org»/ui/«T»ViewModel.kt",
        lines: &[
            " class «T»ViewModel(private val repo: «T»Repository) : ViewModel() {",
            "-    val state = MutableLiveData<«T»>()",
            "+    private val _state = MutableStateFlow<«T»?>(null)",
            "+    val state: StateFlow<«T»?> = _state",
            " ",
            "     fun refresh(id: String) = viewModelScope.launch {",
            "-        state.value = repo.«f»(id)",
            "+        _state.value = repo.«f»(id)",
            "     }",
            " }",
        ] },
    // -- csharp ----------------------------------------------------------------------------
    Template { language: "csharp", name: "async-action",
        path: "src/«Org».Api/Controllers/«T»Controller.cs",
        lines: &[
            " [HttpGet(\"{id}\")]",
            "-public «T» Get(int id)",
            "+public async Task<ActionResult<«T»>> Get(int id)",
            " {",
            "-    return _db.«T»s.Find(id);",
            "+    var entity = await _db.«T»s.FindAsync(id);",
            "+    return entity is null ? NotFound() : entity;",
            " }",
            " ",
            " [HttpDelete(\"{id}\")]",
            " public async Task<IActionResult> Delete(int id)",
        ] },
    Template { language: "csharp", name: "record-member",
        path: "src/«Org».Domain/«T».cs",
        lines: &[
            " namespace «Org».Domain;",
            " ",
            " public sealed record «T»(",
            "     string «A»,",
            "-    int «B»)",
            "+    int «B»,",
            "+    DateTimeOffset UpdatedAt)",
            " {",
            "-    public bool IsEmpty => string.IsNullOrEmpty(«A»);",
            "+    public bool IsExpired(DateTimeOffset now) => now - UpdatedAt > TimeSpan.FromDays(«n»);",
            " }",
        ] },
    Template { language: "csharp", name: "linq-order",
        path: "src/«Org».Services/«T»Query.cs",
        lines: &[
            " public IReadOnlyList<«T»> «F»(IEnumerable<«T»> source)",
            " {",
            "-    return source.Where(x => x.«A» != null).ToList();",
            "+    return source",
            "+        .Where(x => !string.IsNullOrEmpty(x.«A»))",
            "+        .OrderBy(x => x.«B»)",
            "+        .ThenBy(x => x.«A», StringComparer.Ordinal)",
            "+        .ToList();",
            " }",
        ] },
    Template { language: "csharp", name: "using-declaration",
        path: "src/«Org».Import/«T»Reader.cs",
        lines: &[
            " public void «F»(string path)",
            " {",
            "-    var stream = File.OpenRead(path);",
            "+    using var stream = File.OpenRead(path);",
            "     var reader = new StreamReader(stream);",
            "     _«a» = reader.ReadToEnd();",
            "-    stream.Close();",
            "+    _logger.LogDebug(\"read {Length} chars from {Path}\", _«a».Length, path);",
            " }",
        ] },
    Template { language: "csharp", name: "switch-expression",
        path: "src/«Org».Domain/«T»Kind.cs",
        lines: &[
            " private static string Describe(«T»Kind kind)",
            " {",
            "-    switch (kind) { case «T»Kind.Open: return \"«w»\"; default: return \"other\"; }",
            "+    return kind switch",
            "+    {",
            "+        «T»Kind.Open => \"«w»\",",
            "+        «T»Kind.Closed => \"closed\",",
            "+        _ => \"other\",",
            "+    };",
            " }",
        ] },
    Template { language: "csharp", name: "null-guard",
        path: "src/«Org».Services/«T»Service.cs",
        lines: &[
            " public «T»Service(I«T»Store store, ILogger<«T»Service> logger)",
            " {",
            "-    _store = store;",
            "-    _logger = logger;",
            "+    _store = store ?? throw new ArgumentNullException(nameof(store));",
            "+    _logger = logger ?? throw new ArgumentNullException(nameof(logger));",
            " }",
            " ",
            " private readonly I«T»Store _store;",
        ] },
    Template { language: "csharp", name: "http-client-options",
        path: "src/«Org».Api/Startup.cs",
        lines: &[
            " services.AddHttpClient<«T»Client>(client =>",
            " {",
            "     client.BaseAddress = new Uri(configuration[\"«T»:BaseUrl\"]);",
            "-    client.Timeout = TimeSpan.FromSeconds(30);",
            "+    client.Timeout = TimeSpan.FromSeconds(«n»);",
            "+    client.DefaultRequestHeaders.Add(\"X-«T»-Region\", \"«w»\");",
            " });",
            " ",
            " services.AddScoped<I«T»Store, Sql«T»Store>();",
        ] },
    // -- php -------------------------------------------------------------------------------
    Template { language: "php", name: "request-validate",
        path: "app/Http/Controllers/«T»Controller.php",
        lines: &[
            " public function «f»(Request $request): JsonResponse",
            " {",
            "-    $data = $request->all();",
            "+    $data = $request->validate([",
            "+        '«a_»' => 'required|string|max:«m»',",
            "+        '«b_»' => 'nullable|integer',",
            "+    ]);",
            "     $record = «T»::create($data);",
            "     return response()->json($record, 201);",
            " }",
        ] },
    Template { language: "php", name: "null-coalesce",
        path: "src/«T»/helpers.php",
        lines: &[
            " <?php",
            " ",
            " function «f_»(array $options): string",
            " {",
            "-    $region = isset($options['«a_»']) ? $options['«a_»'] : '«w»';",
            "+    $region = $options['«a_»'] ?? '«w»';",
            "     return strtoupper($region);",
            " }",
        ] },
    Template { language: "php", name: "typed-properties",
        path: "src/Domain/«T».php",
        lines: &[
            " final class «T»",
            " {",
            "-    private $«a»;",
            "+    private string $«a»;",
            "+    private int $«b» = «n»;",
            " ",
            "     public function __construct(string $«a»)",
            "     {",
            "         $this->«a» = $«a»;",
            "     }",
        ] },
    Template { language: "php", name: "prepared-statement",
        path: "src/Repository/«T»Repository.php",
        lines: &[
            " public function findBy«A»(string $value): array",
            " {",
            "-    $rows = $this->pdo->query(\"SELECT * FROM «t»s WHERE «a_» = '$value'\");",
            "+    $stmt = $this->pdo->prepare('SELECT * FROM «t»s WHERE «a_» = :value');",
            "+    $stmt->execute(['value' => $value]);",
            "+    $rows = $stmt->fetchAll();",
            "     $out = [];",
            "     foreach ($rows as $row) {",
            "         $out[] = «T»::fromRow($row);",
            "     }",
        ] },
    Template { language: "php", name: "match-expression",
        path: "src/Domain/«T»Status.php",
        lines: &[
            " public function label(): string",
            " {",
            "-    switch ($this->status) {",
            "-        case '«w»': return 'Waiting';",
            "-        default: return 'Unknown';",
            "-    }",
            "+    return match ($this->status) {",
            "+        '«w»' => 'Waiting',",
            "+        default => 'Unknown',",
            "+    };",
            " }",
        ] },
    Template { language: "php", name: "array-column",
        path: "src/Report/«T»Report.php",
        lines: &[
            " $«a»List = [];",
            "-foreach ($«t»s as $«t») {",
            "-    $«a»List[] = $«t»->«a»;",
            "-}",
            "+$«a»List = array_column($«t»s, '«a»');",
            " sort($«a»List);",
            " ",
            " return implode(', ', $«a»List);",
        ] },
    Template { language: "php", name: "catch-specific",
        path: "src/Delivery/«T»Sender.php",
        lines: &[
            " try {",
            "     $this->client->send($«t»);",
            "-} catch (Exception $e) {",
            "-    error_log($e->getMessage());",
            "+} catch (ConnectException $e) {",
            "+    $this->logger->warning('«t» delivery failed', ['error' => $e->getMessage()]);",
            "+    throw new «T»Unavailable($e->getMessage(), 0, $e);",
            " }",
            " ",
            " return true;",
        ] },
    // -- scala -----------------------------------------------------------------------------
    Template { language: "scala", name: "future-option",
        path: "src/main/scala/«org»/«t»/«T»Service.scala",
        lines: &[
            " class «T»Service(repo: «T»Repo)(implicit ec: ExecutionContext) {",
            " ",
            "   def «f»(id: Long): Future[«T»] =",
            "-    repo.find(id).map(_.get)",
            "+    repo.find(id).flatMap {",
            "+      case Some(found) => Future.successful(found)",
            "+      case None        => Future.failed(new NoSuchElementException(s\"«t» $id\"))",
            "+    }",
            " }",
        ] },
    Template { language: "scala", name: "case-class-copy",
        path: "src/main/scala/«org»/model/«T».scala",
        lines: &[
            " package «org».model",
            " ",
            " final case class «T»(«a»: String, «b»: Int) {",
            "-  def bump: «T» = «T»(«a», «b» + 1)",
            "+  def bump: «T» = copy(«b» = «b» + «n»)",
            "+  def renamed(next: String): «T» = copy(«a» = next)",
            " }",
            " ",
            " object «T» { val empty: «T» = «T»(\"\", 0) }",
        ] },
    Template { language: "scala", name: "match-cases",
        path: "src/main/scala/«org»/events/«T»Events.scala",
        lines: &[
            " def describe(event: «T»Event): String = event match {",
            "   case «T»Event.Created(id) => s\"created $id\"",
            "-  case _                    => \"ignored\"",
            "+  case «T»Event.Moved(id, to) => s\"moved $id to $to\"",
            "+  case other                => s\"ignored ${other.getClass.getSimpleName}\"",
            " }",
            " ",
            " def isTerminal(event: «T»Event): Boolean = event.isInstanceOf[«T»Event.Closed]",
        ] },
    Template { language: "scala", name: "actor-receive",
        path: "src/main/scala/«org»/actors/«T»Actor.scala",
        lines: &[
            " class «T»Actor(store: «T»Store) extends Actor with ActorLogging {",
            "   def receive: Receive = {",
            "-    case Fetch(id) => sender() ! store.get(id)",
            "+    case Fetch(id) =>",
            "+      log.debug(\"fetching {}\", id)",
            "+      sender() ! store.get(id).getOrElse(NotFound(id))",
            "     case Put(id, value) => store.put(id, value)",
            "   }",
            " }",
        ] },
    Template { language: "scala", name: "implicit-ordering",
        path: "src/main/scala/«org»/model/«T»Ordering.scala",
        lines: &[
            " package «org».model",
            " ",
            " object «T»Ordering {",
            "-  implicit val ordering: Ordering[«T»] = Ordering.by(_.«a»)",
            "+  implicit val ordering: Ordering[«T»] =",
            "+    Ordering.by[«T», (Int, String)](x => (x.«b», x.«a»))",
            " }",
            " ",
            " final case class «T»(«a»: String, «b»: Int)",
        ] },
    Template { language: "scala", name: "for-comprehension",
        path: "src/main/scala/«org»/flows/«T»Flow.scala",
        lines: &[
            " import scala.concurrent.TimeoutException",
            " ",
            " def latestFor(userId: Long): Future[«T»] =",
            "   for {",
            "     account <- accounts.lookup(userId)",
            "-    «t»     <- «t»s.latest(account)",
            "+    «t»     <- «t»s.latest(account).recover { case _: TimeoutException => «T».empty }",
            "+    _       <- audit.record(userId, «t».id)",
            "   } yield «t»",
        ] },
    Template { language: "scala", name: "lazy-config",
        path: "src/main/scala/«org»/config/«T»Config.scala",
        lines: &[
            " import com.typesafe.config.Config",
            " ",
            " class «T»Config(raw: Config) {",
            "-  val «a»: String = raw.getString(\"«a_»\")",
            "+  lazy val «a»: String = raw.getString(\"«a_»\")",
            "+  lazy val «b»: Int = raw.getInt(\"«b_»\")",
            " ",
            "   def describe: String = s\"«t»(${«a»})\"",
            " }",
        ] },
    // -- elixir ----------------------------------------------------------------------------
    Template { language: "elixir", name: "with-clause",
        path: "lib/«org»/«t»s.ex",
        lines: &[
            " def «f_»(params) do",
            "-  {:ok, «t»} = Repo.insert(«T».changeset(%«T»{}, params))",
            "-  «t»",
            "+  with {:ok, «t»} <- Repo.insert(«T».changeset(%«T»{}, params)) do",
            "+    {:ok, «t»}",
            "+  else",
            "+    {:error, changeset} -> {:error, changeset}",
            "+  end",
            " end",
        ] },
    Template { language: "elixir", name: "genserver-cast",
        path: "lib/«org»/«t»_cache.ex",
        lines: &[
            " def handle_call({:get, key}, _from, state) do",
            "-  {:reply, Map.get(state, key), state}",
            "+  {:reply, Map.fetch(state, key), state}",
            " end",
            " ",
            "+def handle_cast({:put, key, value}, state) do",
            "+  {:noreply, Map.put(state, key, value)}",
            "+end",
            "+",
        ] },
    Template { language: "elixir", name: "pipe-reject",
        path: "lib/«org»/«t»_import.ex",
        lines: &[
            " def «f_»(rows) do",
            "   rows",
            "-  |> Enum.map(&String.downcase/1)",
            "+  |> Enum.map(&(&1 |> String.trim() |> String.downcase()))",
            "+  |> Enum.reject(&(&1 == \"\"))",
            "   |> Enum.uniq()",
            "   |> Enum.sort()",
            " end",
        ] },
    Template { language: "elixir", name: "schema-field",
        path: "lib/«org»/«t».ex",
        lines: &[
            " schema \"«t»s\" do",
            "   field :«a_», :string",
            "-  field :«b_», :integer",
            "+  field :«b_», :integer, default: «n»",
            "+  field :archived_at, :utc_datetime",
            "   belongs_to :«org», «Org».Account",
            " ",
            "   timestamps()",
            " end",
        ] },
    Template { language: "elixir", name: "guards",
        path: "lib/«org»/«t»_rates.ex",
        lines: &[
            " @doc \"«w» rate for one «t»\"",
            "-def «f_»(n) when n > 0, do: n * «n»",
            "+def «f_»(n) when is_integer(n) and n > 0, do: n * «n»",
            "+def «f_»(_), do: {:error, :invalid}",
            " ",
            " @doc false",
            " def default_rate, do: «n»",
            " def rates, do: [«n», «m»]",
        ] },
    Template { language: "elixir", name: "case-status",
        path: "lib/«org»/clients/«t»_client.ex",
        lines: &[
            " def fetch(url) do",
            "   case HTTPoison.get(url) do",
            "     {:ok, %{status_code: 200, body: body}} -> Jason.decode(body)",
            "-    _ -> :error",
            "+    {:ok, %{status_code: code}} -> {:error, {:http, code}}",
            "+    {:error, reason} -> {:error, reason}",
            "   end",
            " end",
        ] },
    Template { language: "elixir", name: "module-attrs",
        path: "lib/«org»/notifiers/«t»_notifier.ex",
        lines: &[
            " defmodule «Org».«T»Notifier do",
            "-  @timeout 5_000",
            "+  @timeout «m»",
            "+  @retries «n»",
            " ",
            "   def deliver(«t»), do: Mailer.deliver(«t», timeout: @timeout)",
            "+  def retries, do: @retries",
            " end",
        ] },
    // -- shell -----------------------------------------------------------------------------
    Template { language: "shell", name: "strict-mode",
        path: "scripts/«t»-sync.sh",
        lines: &[
            " #!/usr/bin/env bash",
            "-set -e",
            "+set -euo pipefail",
            " ",
            " «TU»_DIR=\"${«TU»_DIR:-/var/lib/«t»}\"",
            " mkdir -p \"$«TU»_DIR\"",
            "-cd $«TU»_DIR",
            "+cd \"$«TU»_DIR\"",
            " rsync -a \"$1\" .",
        ] },
    Template { language: "shell", name: "quote-vars",
        path: "scripts/«t»-backup.sh",
        lines: &[
            " SRC=\"$1\"",
            " DEST=\"$2\"",
            " ",
            " for f in \"$SRC\"/*.log; do",
            "-  cp $f $DEST/",
            "+  cp -- \"$f\" \"$DEST/\"",
            " done",
            " ",
            "-echo done",
            "+echo \"«t» backup: $(ls \"$DEST\" | wc -l) files\"",
        ] },
    Template { language: "shell", name: "retry-loop",
        path: "bin/«t»-wait.sh",
        lines: &[
            " «f_»() {",
            "   local attempt=1",
            "-  until curl -fsS \"$1\"; do",
            "+  until curl -fsS --max-time «n» \"$1\"; do",
            "+    [ \"$attempt\" -ge «n» ] && return 1",
            "     attempt=$((attempt + 1))",
            "     sleep 2",
            "   done",
            " }",
        ] },
    Template { language: "shell", name: "getopts",
        path: "scripts/«t»-export.sh",
        lines: &[
            " VERBOSE=0",
            "-while getopts \"v\" opt; do",
            "+OUT_FILE=\"«t».csv\"",
            "+while getopts \"vo:\" opt; do",
            "   case \"$opt\" in",
            "     v) VERBOSE=1 ;;",
            "+    o) OUT_FILE=\"$OPTARG\" ;;",
            "   esac",
            " done",
        ] },
    Template { language: "shell", name: "trap-cleanup",
        path: "scripts/«t»-unpack.sh",
        lines: &[
            " tmp=$(mktemp -d)",
            "+trap 'rm -rf \"$tmp\"' EXIT",
            " tar -xzf \"$1\" -C \"$tmp\"",
            "-«f_» \"$tmp\"",
            "-rm -rf \"$tmp\"",
            "+«f_» \"$tmp\" \"«w»\"",
            " ",
            " echo \"unpacked $1\"",
        ] },
    Template { language: "shell", name: "double-brackets",
        path: "bin/«t»-ctl.sh",
        lines: &[
            " #!/usr/bin/env bash",
            "-if [ $# -lt 1 ]; then",
            "+if [[ $# -lt 1 ]]; then",
            "   echo \"usage: $0 <«t»-id>\" >&2",
            "   exit 64",
            " fi",
            "+readonly «TU»_ID=\"$1\"",
            " ",
            " exec «t»ctl status \"$1\"",
        ] },
    Template { language: "shell", name: "env-defaults",
        path: "deploy/«t»-run.sh",
        lines: &[
            " #!/bin/sh",
            " # «w» profile",
            " export «TU»_HOST=\"${«TU»_HOST:-localhost}\"",
            "-export «TU»_PORT=8080",
            "+export «TU»_PORT=\"${«TU»_PORT:-«m»}\"",
            "+export «TU»_LOG_LEVEL=\"${«TU»_LOG_LEVEL:-info}\"",
            " ",
            " exec «t»-server --host \"$«TU»_HOST\" --port \"$«TU»_PORT\"",
        ] },
];

/// `sha256` over every template and every identifier pool. The allowlist records it, so a
/// catalogue edited after the allowlist was made is refused rather than silently re-split.
pub fn catalogue_sha256() -> String {
    let mut hasher = Sha256::new();
    for t in CATALOGUE {
        for part in [t.language, t.name, t.path] {
            hasher.update(part.as_bytes());
            hasher.update([0u8]);
        }
        for line in t.lines {
            hasher.update(line.as_bytes());
            hasher.update(b"\n");
        }
        hasher.update([1u8]);
    }
    for pool in [TYPES, VERBS, NOUNS, WORDS, ORGS] {
        for word in pool {
            hasher.update(word.as_bytes());
            hasher.update([0u8]);
        }
        hasher.update([1u8]);
    }
    hasher.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

/// Every template unit, sorted. What `qd-noul-rows units` prints for the allowlist step.
pub fn units() -> Vec<String> {
    let mut out: Vec<String> = CATALOGUE.iter().map(Template::unit).collect();
    out.sort();
    out
}

fn pascal(snake: &str) -> String {
    snake
        .split('_')
        .map(|part| {
            let mut chars = part.chars();
            match chars.next() {
                Some(first) => first.to_ascii_uppercase().to_string() + chars.as_str(),
                None => String::new(),
            }
        })
        .collect()
}

fn camel(snake: &str) -> String {
    let p = pascal(snake);
    let mut chars = p.chars();
    match chars.next() {
        Some(first) => first.to_ascii_lowercase().to_string() + chars.as_str(),
        None => String::new(),
    }
}

fn snake_of_pascal(name: &str) -> String {
    let mut out = String::new();
    for (i, c) in name.chars().enumerate() {
        if c.is_ascii_uppercase() && i > 0 {
            out.push('_');
        }
        out.push(c.to_ascii_lowercase());
    }
    out
}

fn pick<'a>(pool: &[&'a str], rng: &mut ChaCha20Rng) -> &'a str {
    pool.choose(rng).copied().unwrap_or(pool[0])
}

/// One draw of every placeholder's value.
fn draw_vars(rng: &mut ChaCha20Rng) -> BTreeMap<&'static str, String> {
    let ty = pick(TYPES, rng);
    let verb = pick(VERBS, rng);
    let a = pick(NOUNS, rng);
    let b = loop {
        let b = pick(NOUNS, rng);
        if b != a {
            break b;
        }
    };
    let org = pick(ORGS, rng);
    let t = snake_of_pascal(ty);
    BTreeMap::from([
        ("T", ty.to_string()),
        ("TU", t.to_ascii_uppercase()),
        ("t", t),
        ("f", camel(verb)),
        ("F", pascal(verb)),
        ("f_", verb.to_string()),
        ("a", camel(a)),
        ("A", pascal(a)),
        ("a_", a.to_string()),
        ("b", camel(b)),
        ("B", pascal(b)),
        ("b_", b.to_string()),
        ("n", rng.random_range(2..=90u32).to_string()),
        ("m", rng.random_range(100..=9_000u32).to_string()),
        ("w", pick(WORDS, rng).to_string()),
        ("org", org.to_string()),
        ("Org", pascal(org)),
    ])
}

/// `text` with every `«key»` replaced. An unknown or unclosed placeholder is an error, never
/// left in a row.
fn fill(text: &str, vars: &BTreeMap<&'static str, String>) -> Result<String, String> {
    let mut out = String::with_capacity(text.len() + 16);
    let mut rest = text;
    while let Some(open) = rest.find('«') {
        out.push_str(&rest[..open]);
        let after = &rest[open + '«'.len_utf8()..];
        let close = after
            .find('»')
            .ok_or_else(|| format!("unclosed placeholder in {text:?}"))?;
        let key = &after[..close];
        let value = vars
            .get(key)
            .ok_or_else(|| format!("unknown placeholder «{key}» in {text:?}"))?;
        out.push_str(value);
        rest = &after[close + '»'.len_utf8()..];
    }
    out.push_str(rest);
    Ok(out)
}

/// `(path, diff)` for one instantiation of `t`.
pub fn instantiate(t: &Template, rng: &mut ChaCha20Rng) -> Result<(String, String), String> {
    let vars = draw_vars(rng);
    let path = fill(t.path, &vars)?;
    let mut lines = Vec::with_capacity(t.lines.len());
    for raw in t.lines {
        let mut chars = raw.chars();
        let mark = chars
            .next()
            .and_then(Mark::from_prefix)
            .ok_or_else(|| format!("{}: line {raw:?} has no diff marker", t.unit()))?;
        lines.push(Line { mark, text: fill(chars.as_str(), &vars)? });
    }
    Ok((path, hunk::render(rng.random_range(5..=600u32), &lines)))
}

pub struct TemplateRow {
    pub unit: String,
    pub template: String,
    pub language: &'static str,
    pub path: String,
    pub diff: String,
}

/// `n` rows, an equal share per language (the first `n % languages` languages, in name order,
/// take one more), round-robin over each language's allowed units. A language with no allowed
/// unit, or too few distinct instantiations, is an error rather than a thinner share.
pub fn rows(
    allowed: &BTreeSet<String>,
    n: usize,
    seed: u64,
) -> Result<(Vec<TemplateRow>, BTreeMap<String, u64>), String> {
    let mut by_language: BTreeMap<&str, Vec<&Template>> = BTreeMap::new();
    for t in CATALOGUE {
        by_language.entry(t.language).or_default();
    }
    for t in CATALOGUE.iter().filter(|t| allowed.contains(&t.unit())) {
        by_language.entry(t.language).or_default().push(t);
    }
    if let Some((lang, _)) = by_language.iter().find(|(_, v)| v.is_empty()) {
        return Err(format!("language {lang} has no train-split template unit in the allowlist"));
    }
    let n_lang = by_language.len();
    let mut out = Vec::with_capacity(n);
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut skipped: BTreeMap<String, u64> = BTreeMap::new();
    for (k, (lang, mut templates)) in by_language.into_iter().enumerate() {
        templates.sort_by_key(|t| t.name);
        let quota = n / n_lang + usize::from(k < n % n_lang);
        let mut got = 0usize;
        let mut i = 0usize;
        while got < quota {
            if i >= quota.saturating_mul(50).max(50) {
                return Err(format!(
                    "{lang}: {got} distinct instantiations after {i} draws; {quota} were asked for"
                ));
            }
            let t = templates[i % templates.len()];
            let mut rng = item_rng(seed, SOURCE, &format!("{}#{i}", t.unit()));
            i += 1;
            let (path, diff) = instantiate(t, &mut rng)?;
            if !seen.insert(diff.clone()) {
                *skipped.entry("duplicate_instantiation".to_string()).or_insert(0) += 1;
                continue;
            }
            out.push(TemplateRow {
                unit: t.unit(),
                template: format!("{}/{}", t.language, t.name),
                language: t.language,
                path,
                diff,
            });
            got += 1;
        }
    }
    Ok((out, skipped))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_template_fills_completely_and_reads_as_one_corpus_shaped_hunk() {
        for t in CATALOGUE {
            for s in 0..5 {
                let mut rng = item_rng(s, "test", t.name);
                let (path, diff) = instantiate(t, &mut rng).unwrap();
                assert!(!path.contains('«') && !diff.contains('«'), "{diff}");
                assert!(!path.contains('»') && !diff.contains('»'), "{diff}");
                assert_eq!(diff.matches("\n@@").count(), 0, "one hunk: {diff}");
                let header = diff.lines().next().unwrap();
                assert!(header.starts_with("@@ -") && header.ends_with(" @@"), "{header}");
                let body: Vec<&str> = diff.lines().skip(1).collect();
                assert!(body.len() >= 7, "{}: {} lines", t.unit(), body.len());
                assert!(body.iter().any(|l| l.starts_with('-') || l.starts_with('+')));
                assert!(body.iter().any(|l| l.starts_with(' ')), "{}", t.unit());
            }
        }
    }

    #[test]
    fn the_catalogue_is_six_languages_seven_units_each_and_unit_names_are_unique() {
        let units = units();
        assert_eq!(units.len(), CATALOGUE.len());
        assert_eq!(units.iter().collect::<BTreeSet<_>>().len(), units.len());
        let mut per: BTreeMap<&str, usize> = BTreeMap::new();
        for t in CATALOGUE {
            *per.entry(t.language).or_insert(0) += 1;
        }
        assert_eq!(
            per,
            BTreeMap::from([("csharp", 7), ("elixir", 7), ("kotlin", 7), ("php", 7),
                            ("scala", 7), ("shell", 7)])
        );
        assert_eq!(catalogue_sha256(), catalogue_sha256());
        assert_eq!(catalogue_sha256().len(), 64);
    }

    #[test]
    fn rows_take_only_allowed_units_split_evenly_and_never_repeat_a_context() {
        let allowed: BTreeSet<String> = units()
            .into_iter()
            .filter(|u| !u.ends_with("/getopts") && !u.ends_with("/guards"))
            .collect();
        let (rows, _) = rows(&allowed, 62, 0).unwrap();
        assert_eq!(rows.len(), 62);
        assert!(rows.iter().all(|r| allowed.contains(&r.unit)));
        let mut per: BTreeMap<&str, usize> = BTreeMap::new();
        for r in &rows {
            *per.entry(r.language).or_insert(0) += 1;
        }
        // 62 over 6 languages: the first two in name order take 11.
        assert_eq!(per["csharp"], 11);
        assert_eq!(per["elixir"], 11);
        assert_eq!(per["shell"], 10);
        let diffs: BTreeSet<&str> = rows.iter().map(|r| r.diff.as_str()).collect();
        assert_eq!(diffs.len(), rows.len());

        let no_shell: BTreeSet<String> =
            units().into_iter().filter(|u| !u.contains("/shell/")).collect();
        assert!(rows_err(&no_shell).contains("shell has no train-split"));
    }

    fn rows_err(allowed: &BTreeSet<String>) -> String {
        match rows(allowed, 12, 0) {
            Ok(_) => String::from("no error"),
            Err(e) => e,
        }
    }

    #[test]
    fn names_convert_between_cases() {
        assert_eq!(pascal("due_date"), "DueDate");
        assert_eq!(camel("due_date"), "dueDate");
        assert_eq!(snake_of_pascal("Greenhouse"), "greenhouse");
        assert!(fill("«nope»", &BTreeMap::new()).unwrap_err().contains("unknown placeholder"));
        assert!(fill("«T", &BTreeMap::new()).unwrap_err().contains("unclosed"));
    }
}
